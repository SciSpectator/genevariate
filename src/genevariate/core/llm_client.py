"""Backend-neutral local-LLM chat client - Ollama (default) or OpenAI-compatible."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional


def _backend() -> str:
    return os.environ.get("LLM_BACKEND", "ollama").strip().lower() or "ollama"


def is_openai_compatible() -> bool:
    """True for vLLM / SGLang / any OpenAI-style backend."""
    return _backend() in ("vllm", "openai", "sglang")


def ollama_url() -> str:
    """Root of the Ollama server, whatever backend ``LLM_BACKEND`` names."""
    url = os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST")
    if not url:
        return "http://localhost:11434"
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


def root_url() -> str:
    """Server root for the active backend, without the endpoint path."""
    if is_openai_compatible():
        url = (
            os.environ.get("VLLM_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("SGLANG_URL")
            or "http://localhost:8000/v1"
        )
        return url.rstrip("/")
    return ollama_url()


def base_url() -> str:
    """Resolved chat-completions URL for the active backend."""
    root = root_url()
    return root + ("/chat/completions" if is_openai_compatible() else "/api/chat")


def default_model() -> str:
    """Model tag the assistant should ask the active backend for."""
    return (os.environ.get("GENEVARIATE_LLM_MODEL")
            or os.environ.get("PHASE1_MODEL")
            or "gemma4:e2b")


def available(model: Optional[str] = None, timeout: float = 3.0) -> tuple:
    """``(ready, reason)`` for the backend ``chat`` would actually call."""
    model = model or default_model()
    root = root_url()
    probe = root + "/models" if is_openai_compatible() else root + "/api/tags"
    try:
        req = urllib.request.Request(probe, method="GET")
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY")
        if api_key and is_openai_compatible():
            req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            json.JSONDecodeError, ValueError) as exc:
        return False, f"no LLM server answered at {root} ({exc})."

    if is_openai_compatible():
        names = [str(m.get("id", "")) for m in (payload.get("data") or [])]
    else:
        names = [str(m.get("name", "")) for m in (payload.get("models") or [])]
    if names and model not in names and f"{model}:latest" not in names:
        return False, f"model {model!r} is not loaded on {root}."
    return True, ""


def _post(url: str, body: Dict, timeout: int) -> Dict:
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json",
               "Accept":       "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY")
    if api_key and "/v1" in url:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def chat(messages: List[Dict[str, str]],
         *,
         model: str,
         temperature: float = 0.0,
         seed: int = 42,
         num_predict: int = -1,
         num_ctx: Optional[int] = None,
         think: Optional[bool] = None,
         keep_alive: int = -1,
         timeout: int = 180,
         retries: int = 3) -> str:
    """Single chat-completion call. Returns the assistant text, or "" on failure."""
    url = base_url()
    if is_openai_compatible():
        body: Dict = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "seed":        seed,
            "stream":      False,
        }
        body["max_tokens"] = 16384 if num_predict in (-1, None) else int(num_predict)
    else:
        opts: Dict = {"temperature": temperature, "seed": seed,
                      "num_predict": num_predict}
        if num_ctx is not None:
            opts["num_ctx"] = int(num_ctx)
        body = {
            "model":      model,
            "messages":   messages,
            "options":    opts,
            "stream":     False,
            "keep_alive": keep_alive,
        }
        if think is not None:
            body["think"] = bool(think)

    last_err: Optional[BaseException] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            data = _post(url, body, timeout=timeout)
            if is_openai_compatible():
                choices = data.get("choices") or []
                if not choices:
                    return ""
                msg = (choices[0] or {}).get("message", {}) or {}
                return (msg.get("content") or "").strip()
            # Ollama
            msg = data.get("message", {}) or {}
            content = (msg.get("content") or "").strip()
            if think:
                thinking = (msg.get("thinking") or "").strip()
                if thinking:
                    return content + "\n" + thinking
            return content
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(2 * attempt)
    if last_err is not None and os.environ.get("LLM_BACKEND_VERBOSE"):
        print(f"[llm_client] {url}: failed after {retries} attempts - {last_err!r}",
              flush=True)
    return ""


__all__ = ["chat", "base_url", "root_url", "ollama_url",
           "is_openai_compatible", "default_model",
           "available"]
