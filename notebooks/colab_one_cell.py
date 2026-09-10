"""Source of notebooks/genevariate_colab.ipynb - edit here, not the JSON.

One Colab cell: serve both extractor checkpoints and publish the endpoint.

Paste the whole file into a single Colab cell and run it. It is idempotent --
rerunning reuses whatever is already healthy instead of reloading 34 GB of
weights, so it is safe to run again after any failure.

It does, in order: install and pin the dependency versions vLLM needs, fetch
the two checkpoints, serve them one at a time sized from the memory actually
free, start the counting router, open a tunnel, measure real latency through
the public URL, and print the exports for the local machine.

No model is substituted and nothing is quantised: the prompt artifacts were
optimised against these checkpoints at bf16.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

M12B = "google/gemma-4-12B-it"
ME2B = "google/gemma-4-E2B-it"
N12B, NE2B = "google/gemma-4-12b-it", "google/gemma-4-e2b-it"
WALL = 100.0          # cloudflare gives up here; slower answers arrive as ""


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.returncode and check:
        sys.exit(f"failed: {cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def port_owners(port):
    """PIDs holding a listening socket on ``port``, read straight from /proc.

    A command-line match is not enough: uvicorn started with --workers forks a
    child whose cmdline is a multiprocessing spawn line, so pkill on
    "uvicorn router:app" reaps the parent while the child keeps the inherited
    listening socket, and the next run dies on "address already in use". The
    socket owner is the only thing that identifies it.
    """
    want = f"{port:04X}"
    inodes = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        f = line.split()
        if f[1].split(":")[1] == want and f[3] == "0A":      # 0A = LISTEN
            inodes.add(f[9])
    if not inodes:
        return []
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                if os.readlink(fd)[8:-1] in inodes:          # socket:[12345]
                    pids.append(int(proc.name))
                    break
        except OSError:
            continue
    return pids


def free_port(port, tries=20):
    """Stop whatever is listening on ``port`` and wait until it lets go."""
    for _ in range(tries):
        pids = port_owners(port)
        if not pids:
            return
        print(f"  port {port} still held by {pids}, stopping them", flush=True)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        time.sleep(1)
    sys.exit(f"port {port} is held by {port_owners(port)} and will not release. "
             f"Runtime -> Restart session, then run the cell again.")


def healthy(port):
    import httpx
    try:
        return httpx.get(f"http://127.0.0.1:{port}/health",
                         timeout=2).status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------- 1. deps
print("=" * 70, "\n1/7  dependencies\n", "=" * 70)
if not Path("/content/.gv_deps_done").exists():
    sh("pip install -q -U vllm huggingface_hub httpx fastapi uvicorn")
    # vLLM upgrades torch to cu130 but does not depend on torchaudio, so
    # Colab's cu128 build is left behind and transformers' audio_utils import
    # raises on the version check. Nothing here uses audio.
    sh("pip uninstall -y -q torchaudio", check=False)
    # transformers 5.15 made head_dim per-layer. gemma-4 really is
    # heterogeneous (256 on sliding layers, 512 global), so the config refuses
    # a single global value while vLLM's convertor still asks for one
    # (vllm-project/vllm#51744). Do NOT unlock it with
    # allow_global_per_layer_attribute_access: that boots with the wrong
    # attention geometry.
    sh('pip install -q "transformers==5.14.1"')
    Path("/content/.gv_deps_done").touch()
else:
    print("already installed (delete /content/.gv_deps_done to redo)")

v = subprocess.run(["vllm", "--version"], capture_output=True, text=True)
if v.returncode:
    sys.exit("vllm cannot import:\n" + v.stdout + v.stderr)
print("vllm", v.stdout.strip())

import httpx                                             # noqa: E402

# ---------------------------------------------------------------- 2. GPU
print("=" * 70, "\n2/7  GPU\n", "=" * 70)
# Deliberately NOT torch: touching torch.cuda here creates a CUDA context in
# the notebook process that holds ~416 MB for as long as the tab is open. That
# is memory the two servers need, and on a 40 GB card it was the difference
# between the E2B starting and dying 256 MB short. nvidia-smi reads the same
# numbers from outside the driver.
QUERY = ("nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap "
         "--format=csv,noheader,nounits")


def gpu():
    out = sh(QUERY, check=False).strip().splitlines()
    if not out:
        sys.exit("No GPU. Runtime -> Change runtime type -> GPU.")
    name, total, free, cap = [x.strip() for x in out[0].split(",")]
    return name, float(total) / 1024, float(free) / 1024, cap


def free_gb():
    return gpu()[2]


NAME, TOTAL, _f, CAP = gpu()
print(f"{NAME}  compute {CAP}  {TOTAL:.2f} GiB")

W12, WE2 = 11.96 * 2, 5.12 * 2            # bf16 bytes/param -> GB of weights
if TOTAL < W12 + WE2 + 3.0:
    sys.exit(
        f"{NAME} ({TOTAL:.0f} GB) cannot hold {W12:.0f} GB + {WE2:.0f} GB "
        f"of bf16 weights plus KV cache.\nThis script will not substitute a "
        f"smaller model or quantise the weights. Use an A100 80 GB or H100.")
# 4096 was arbitrary and it is what starved the card. Measured with the real
# gemma-4-12B tokenizer over all 3,442 GPL24676 samples, the worst phase-1
# input is 1041 tokens (largest prompt 663 + largest metadata block 378) and
# phase 2 is smaller still, so 2048 leaves 750 tokens of headroom on the worst
# sample in the corpus. Nothing is truncated; the shorter length simply frees
# the activation memory that a 4096 profiling pass reserves.
MAX_LEN = 2048

# ---------------------------------------------------------------- 3. weights
print("=" * 70, "\n3/7  weights\n", "=" * 70)
from huggingface_hub import snapshot_download            # noqa: E402
for repo in (M12B, ME2B):
    print("fetching", repo, flush=True)
    snapshot_download(repo)
print("weights ready")

# ---------------------------------------------------------------- 4. serve
print("=" * 70, "\n4/7  serve\n", "=" * 70)


CAUSE = re.compile(
    r"ValueError|RuntimeError|torch\.OutOfMemoryError|CUDA out of memory|"
    r"KV cache|max seq len|free memory|No available memory|Error |ERROR")


def report(log):
    """Print vLLM's real complaint, which sits above the outer traceback."""
    lines = open(log).readlines()
    hits = [i for i, ln in enumerate(lines) if CAUSE.search(ln)]
    if hits:
        lo = max(0, hits[0] - 3)
        print("  --- root cause ---")
        print("".join(lines[lo:hits[0] + 12]))
    print("  --- last 20 lines ---")
    print("".join(lines[-20:]))
    text = "".join(lines)
    if "max seq len" in text and "KV cache" in text:
        print(f"  -> The KV cache cannot hold one sequence of {MAX_LEN} "
              f"tokens. Set MAX_LEN lower in the cell and run it again, or "
              f"move to an 80 GB card.")


def kv_tokens(log):
    """How many tokens the server actually got, straight from its own log."""
    m = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", open(log).read())
    return int(m[-1].replace(",", "")) if m else 0


def serve(port, repo, alias, frac, extra, minutes=25):
    """Start one vLLM server against its share of the card."""
    log = f"/content/vllm_{port}.log"
    if healthy(port):
        print(f":{port} already healthy, reusing")
        return kv_tokens(log) or 1
    print(f"  fraction {frac:.3f} = {frac * TOTAL:.2f} GB")

    cmd = ["vllm", "serve", repo, "--port", str(port),
           "--served-model-name", alias, repo,
           "--gpu-memory-utilization", f"{frac:.3f}",
           "--max-model-len", str(MAX_LEN),
           "--dtype", "bfloat16",
           # No CUDA graphs: capture needs memory on top of weights and KV,
           # and that is exactly what runs out here. Eager is slower but
           # numerically identical.
           "--enforce-eager",
           *extra]
    print("$", " ".join(cmd), flush=True)
    # vLLM's own advice after it ran 256 MB short: allocating in expandable
    # segments keeps the tail end of a nearly full card from fragmenting.
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < minutes * 60:
        if proc.poll() is not None:
            print(f"  exited code {proc.returncode} after {time.time()-t0:.0f}s")
            report(log)
            return 0
        if healthy(port):
            tok = kv_tokens(log)
            print(f"  ready in {time.time()-t0:.0f}s, KV cache {tok:,} tokens "
                  f"= {tok / MAX_LEN:.1f} concurrent sequences")
            return tok or 1
        print(f"  {time.time()-t0:5.0f}s loading {repo}", flush=True)
        time.sleep(20)
    print("  timed out")
    report(log)
    return 0


CTX_GB = 1.5      # measured: a server occupies its budget plus this much
SLACK_GB = 0.3
# vLLM's budget must cover weights AND the activation peak it reserves while
# profiling, before any KV cache is allocated. Leaving activations out is what
# starved the 12B: a 24.66 GB budget minus 23.92 GB of weights left less than
# the activation peak, so the KV cache came out negative and the engine
# refused to start. Measured on a 40 GB A100: the E2B's peak was 0.42 GB.
ACT12, ACTE2 = 1.0, 0.45
# Bytes per cached token from each config.json: 12B has 40 sliding layers
# (window 1024, 8 kv heads x 256) and 8 full-attention layers (1 x 512); the
# E2B has 28 sliding (window 512) and 7 full, with 1 kv head.
KVSEQ12 = 0.344 if MAX_LEN <= 2048 else 0.375
KVSEQE2 = 0.041 if MAX_LEN <= 2048 else 0.068
PLAN = Path("/content/.gv_plan.json")


def make_plan():
    """Split the card between the two servers before either is loaded."""
    avail = free_gb()
    if avail < 45:
        # A 40 GB card has no room to optimise: weights, activations and two
        # CUDA contexts already claim ~39 GB of it. This is the one split
        # observed to get both engines past memory profiling on this card, so
        # it is used as measured rather than recomputed.
        kv12 = 0.652 * TOTAL - W12 - ACT12
        if kv12 < KVSEQ12:
            sys.exit(f"{NAME} is too small for both models at bf16. "
                     f"Use an 80 GB card.")
        print(f"{avail:.2f} GB free - using the split measured on a 40 GB "
              f"A100 (0.652 / 0.284); 12B KV {kv12:.2f} GB.\n  Together these "
              f"claim the whole card, which is why CUDA graphs are disabled.")
        return {"frac12": 0.652, "frace2": 0.284, "kv12": kv12}

    pool = avail - 2 * CTX_GB - W12 - WE2 - ACT12 - ACTE2 - SLACK_GB
    print(f"{avail:.2f} GB free - {2 * CTX_GB:.1f} context - {W12 + WE2:.2f} "
          f"weights - {ACT12 + ACTE2:.2f} activations - {SLACK_GB} slack = "
          f"{pool:.2f} GB for KV")
    if pool < KVSEQ12 + KVSEQE2:
        sys.exit(
            f"{NAME} leaves {pool:.2f} GB for KV cache, less than the "
            f"{KVSEQ12 + KVSEQE2:.2f} GB one sequence each costs.\nThis "
            f"script will not quantise the weights or drop a model to make it "
            f"fit, because that changes the labels. Use an 80 GB card.")
    # The 12B answers every sample; the E2B is only asked for Age and phase 2.
    kv12, kve2 = 0.6 * pool, 0.4 * pool
    return {"frac12": (W12 + ACT12 + kv12) / TOTAL,
            "frace2": (WE2 + ACTE2 + kve2) / TOTAL, "kv12": kv12}


# The budgets are decided once, while the card is empty: a server already
# running would hide the memory it holds, so a fresh measurement would size the
# next one against a card that looks far smaller than it is. When both servers
# survive from an earlier run the plan is read back instead of recomputed.
if healthy(8000) and healthy(8001) and PLAN.exists():
    plan = json.loads(PLAN.read_text())
    print("both servers already healthy; reusing the recorded plan")
else:
    sh('pkill -f "vllm serve" || true', check=False)
    # A killed server hands its memory back asynchronously, so wait for the
    # reading to stop rising rather than guessing a sleep.
    prev = -1.0
    for _ in range(20):
        time.sleep(3)
        now = free_gb()
        if now <= prev + 0.05:
            break
        prev = now
    plan = make_plan()
    PLAN.write_text(json.dumps(plan))

F12, FE2 = plan["frac12"], plan["frace2"]
print(f"12B -> fraction {F12:.3f} ({F12 * TOTAL:.2f} GB)   "
      f"E2B -> fraction {FE2:.3f} ({FE2 * TOTAL:.2f} GB)")

TOK12 = serve(8000, M12B, N12B, F12, [])
if not TOK12:
    sys.exit("the 12B did not come up - see the log above")
TOKE2 = serve(8001, ME2B, NE2B, FE2, ["--reasoning-parser", "gemma4"])
if not TOKE2:
    sys.exit("the E2B did not come up - see the log above")

# ---------------------------------------------------------------- 5. router
print("=" * 70, "\n5/7  router\n", "=" * 70)
import secrets                                           # noqa: E402
API_KEY = secrets.token_urlsafe(24)

Path("/content/router.py").write_text('''
"""Name-based reverse proxy: one OpenAI base URL, both extractor models.

Counts what it answered. A proxy cut-off is invisible from both ends -- the
router logs a 200 while the client already read "" -- so answered-vs-received
is the only evidence that requests were dropped.
"""
import os, time
from collections import defaultdict
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

UPSTREAM = {
    "google/gemma-4-12b-it": "http://127.0.0.1:8000",
    "google/gemma-4-12B-it": "http://127.0.0.1:8000",
    "google/gemma-4-e2b-it": "http://127.0.0.1:8001",
    "google/gemma-4-E2B-it": "http://127.0.0.1:8001",
}
API_KEY = os.environ.get("GV_API_KEY", "")
WALL = float(os.environ.get("GV_WALL_SECONDS", "100"))

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(900.0),
                           limits=httpx.Limits(max_connections=256))
S = {k: defaultdict(int) for k in
     ("received", "answered", "upstream_error", "over_wall", "unknown_model")}
S["seconds"] = defaultdict(float)
S["max_seconds"] = defaultdict(float)
S["unauthorized"] = 0
S["started"] = time.time()


def _denied(request):
    if not API_KEY:
        return None
    if request.headers.get("authorization", "") == f"Bearer {API_KEY}":
        return None
    S["unauthorized"] += 1
    return JSONResponse({"error": {"message": "bad or missing bearer token",
                                   "type": "invalid_request_error"}},
                        status_code=401)


@app.get("/health")
async def health():
    return {"status": "ok", "models": sorted(UPSTREAM)}


@app.get("/_stats")
async def stats(request: Request):
    bad = _denied(request)
    if bad is not None:
        return bad
    names = set()
    for k in ("received", "answered", "upstream_error", "over_wall",
              "unknown_model"):
        names |= set(S[k])
    out = {"uptime_seconds": round(time.time() - S["started"], 1),
           "wall_seconds": WALL, "unauthorized": S["unauthorized"],
           "models": {}}
    for n in sorted(names):
        a = S["answered"][n]
        out["models"][n] = {
            "received": S["received"][n], "answered": a,
            "upstream_error": S["upstream_error"][n],
            "unknown_model": S["unknown_model"][n],
            "over_wall": S["over_wall"][n],
            "mean_seconds": round(S["seconds"][n] / a, 2) if a else None,
            "max_seconds": round(S["max_seconds"][n], 2)}
    return out


@app.get("/v1/models")
async def models(request: Request):
    bad = _denied(request)
    if bad is not None:
        return bad
    return {"object": "list",
            "data": [{"id": n, "object": "model", "owned_by": "google"}
                     for n in sorted(UPSTREAM)]}


@app.post("/v1/{path:path}")
async def forward(path: str, request: Request):
    bad = _denied(request)
    if bad is not None:
        return bad
    body = await request.json()
    name = body.get("model", "")
    S["received"][name] += 1
    target = UPSTREAM.get(name)
    if target is None:
        S["unknown_model"][name] += 1
        return JSONResponse(
            {"error": {"message": f"model {name!r} is not served here",
                       "type": "invalid_request_error"}}, status_code=404)
    t0 = time.time()
    try:
        r = await client.post(f"{target}/v1/{path}", json=body)
    except Exception as exc:
        S["upstream_error"][name] += 1
        return JSONResponse({"error": {"message": f"upstream failed: {exc}",
                                       "type": "api_error"}}, status_code=502)
    dt = time.time() - t0
    S["answered"][name] += 1
    S["seconds"][name] += dt
    S["max_seconds"][name] = max(S["max_seconds"][name], dt)
    if dt > WALL:
        S["over_wall"][name] += 1
    return JSONResponse(r.json(), status_code=r.status_code)
''')

sh('pkill -f "uvicorn router:app" || true', check=False)
free_port(9000)
# No --workers: one worker through the supervisor buys nothing and forks a
# child that outlives pkill still holding port 9000. In-process, the router is
# the one thing that owns the socket.
subprocess.Popen(["uvicorn", "router:app", "--host", "0.0.0.0", "--port",
                  "9000"],
                 cwd="/content", env=dict(os.environ, GV_API_KEY=API_KEY),
                 stdout=open("/content/router.log", "w"),
                 stderr=subprocess.STDOUT)
for _ in range(90):
    if healthy(9000):
        break
    time.sleep(1)
else:
    sys.exit("router did not start:\n" + open("/content/router.log").read())
print("router up")

H = {"Authorization": f"Bearer {API_KEY}"}
r = httpx.post("http://127.0.0.1:9000/v1/chat/completions", headers=H,
               json={"model": N12B, "temperature": 0.0, "max_tokens": 8,
                     "messages": [{"role": "user",
                                   "content": "Reply with one word: ok"}]},
               timeout=300)
print("12B ->", r.status_code, r.json()["choices"][0]["message"].get("content"))

r = httpx.post("http://127.0.0.1:9000/v1/chat/completions", headers=H,
               json={"model": NE2B, "temperature": 0.0, "max_tokens": 256,
                     "chat_template_kwargs": {"enable_thinking": True},
                     "messages": [{"role": "user",
                                   "content": "A donor is 47 years old. "
                                              "Reply with just the number."}]},
               timeout=300)
msg = r.json()["choices"][0]["message"]
print("E2B ->", r.status_code, "reasoning_content:",
      bool(msg.get("reasoning_content")))
if "reasoning_content" not in msg:
    sys.exit("the reasoning parser is not active on :8001 -- phase 2 falls "
             "back to reasoning_content when content is empty, so labels "
             "would be dropped silently.")

# ---------------------------------------------------------------- 6. tunnel
print("=" * 70, "\n6/7  public endpoint\n", "=" * 70)
if not Path("/usr/local/bin/cloudflared").exists():
    sh("wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/"
       "cloudflared/releases/latest/download/cloudflared-linux-amd64 "
       "&& chmod +x /usr/local/bin/cloudflared")
sh('pkill -f "cloudflared tunnel" || true', check=False)
time.sleep(2)
subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--url",
                  "http://localhost:9000"],
                 stdout=open("/content/cf.log", "w"),
                 stderr=subprocess.STDOUT)
PUBLIC_URL = None
for _ in range(90):
    time.sleep(1)
    m = re.search(r"https://[a-z0-9.-]+\.trycloudflare\.com",
                  open("/content/cf.log").read())
    if m:
        PUBLIC_URL = m.group(0)
        break
if not PUBLIC_URL:
    sys.exit("tunnel failed:\n" + open("/content/cf.log").read()[-2000:])
print("public URL:", PUBLIC_URL)

# ---------------------------------------------------------------- 7. probe
print("=" * 70, "\n7/7  latency probe\n", "=" * 70)
# Asking for more concurrency than the KV cache holds only builds a queue, and
# a request that waits past the tunnel's 100 s wall comes back as "", which the
# pipeline reads as Not Specified. These are the cache sizes the two servers
# reported for themselves, so the limit is measured rather than predicted.
SEQS = min(TOK12, TOKE2) / MAX_LEN
WORKERS = max(1, min(8, int(SEQS)))
KNOBS = dict(p1=256, p2=384, ceil=1536, think=192, workers=WORKERS,
             ctx=MAX_LEN)
print(f"12B {TOK12:,} tok, E2B {TOKE2:,} tok, {MAX_LEN} per "
      f"sequence -> {SEQS:.1f} concurrent -> {WORKERS} workers")
SAMPLE = ("title: Liver biopsy from hepatocellular carcinoma patient\n"
          "source_name: liver tumor tissue\n"
          "characteristics: tissue: liver; disease: hepatocellular carcinoma; "
          "Sex: male; age: 62 years")
SHAPES = {
    "phase1 (12B)": dict(
        model=N12B, temperature=0.0, max_tokens=KNOBS["p1"],
        messages=[{"role": "user",
                   "content": "Extract Tissue, Condition, Treatment and Sex "
                              "from this GEO sample as JSON.\n\n" + SAMPLE}]),
    "phase2 (E2B)": dict(
        model=NE2B, temperature=0.0, max_tokens=KNOBS["p2"],
        chat_template_kwargs={"enable_thinking": True},
        messages=[{"role": "user",
                   "content": 'Normalise "hepatocellular carcinoma of the '
                              'liver" to one MeSH preferred term.'}]),
}
BURST = max(2 * KNOBS["workers"], 8)


def one(body):
    t0 = time.time()
    try:
        r = httpx.post(PUBLIC_URL + "/v1/chat/completions", headers=H,
                       json=body, timeout=180)
        m = r.json()["choices"][0]["message"] if r.status_code == 200 else {}
        return (time.time() - t0, r.status_code,
                not (m.get("content") or m.get("reasoning_content")))
    except Exception as exc:
        return time.time() - t0, type(exc).__name__, True


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


before = httpx.get("http://127.0.0.1:9000/_stats", headers=H, timeout=30).json()
print(f"{BURST} requests per shape at concurrency {KNOBS['workers']}\n")
ok = True
for label, body in SHAPES.items():
    with ThreadPoolExecutor(max_workers=KNOBS["workers"]) as ex:
        res = list(ex.map(one, [dict(body)] * BURST))
    lat = [d for d, _, _ in res]
    bad = [c for _, c, _ in res if c != 200]
    empty = sum(1 for _, _, e in res if e)
    over = sum(1 for d in lat if d > WALL)
    print(f"{label:<14} p50 {pct(lat,.5):6.1f}s  p95 {pct(lat,.95):6.1f}s  "
          f"max {max(lat):6.1f}s   non-200 {len(bad)}  empty {empty}  "
          f"over-{WALL:.0f}s {over}")
    if bad:
        print(f"{'':<14} statuses {sorted(set(map(str, bad)))}")
    if bad or empty or over or pct(lat, .95) > 0.6 * WALL:
        ok = False

after = httpx.get("http://127.0.0.1:9000/_stats", headers=H, timeout=30).json()
served = (sum(v["answered"] for v in after["models"].values())
          - sum(v["answered"] for v in before["models"].values()))
print(f"\nrouter answered {served} / {BURST * len(SHAPES)} sent")
print("\nVERDICT:", "OK - budgets fit" if ok else
      "NOT OK - answers are near or past the 100 s wall; lowering "
      "PHASE1_MAX_TOKENS / GEO_EXTRACT_WORKERS is required or labels will "
      "silently come back as Not Specified")

print("\n" + "=" * 70)
print("# paste into the shell that runs GeneVariate")
print('export LLM_BACKEND=vllm')
print(f'export VLLM_URL="{PUBLIC_URL}/v1"')
print(f'export OPENAI_API_KEY="{API_KEY}"')
print(f'export PHASE1_MODEL={N12B}')
print(f'export AGE_MODEL={NE2B}')
print(f'export PHASE2_MODEL={NE2B}')
print(f'export PHASE1_MAX_TOKENS={KNOBS["p1"]}')
print(f'export PHASE2_MAX_TOKENS={KNOBS["p2"]}')
print(f'export PHASE2_TOKEN_CEILING={KNOBS["ceil"]}')
print(f'export PHASE2_THINK_BUDGET={KNOBS["think"]}')
print(f'export LLM_NUM_CTX={KNOBS["ctx"]}')
print(f'export GEO_EXTRACT_WORKERS={KNOBS["workers"]}')
print("=" * 70)
print("\nKeep this tab open: closing it kills the GPU and the URL.")
