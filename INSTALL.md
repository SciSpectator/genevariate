# GeneVariate — Installation Guide

Pick the section that matches your setup.

- [Linux (Ubuntu / Debian / Fedora / Arch)](#linux)
- [macOS](#macos)
- [Windows](#windows)
- [Docker (all platforms)](#docker)
- [After install: models + GEOmetadb](#post-install)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

Every path below ends at the same place: `genevariate` running on your machine, with local
Ollama models and the GEOmetadb SQLite catalogue.

---

## Linux

### Ubuntu / Debian

```bash
# 1. System dependencies
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-tk git-lfs

# 2. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 3. Clone and enter the repo
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate

# 4. Create a virtualenv and install
python3 -m venv venv
source venv/bin/activate
pip install -e ".[analysis]"

# 5. Desktop launcher (optional)
python3 install.py
```

### Fedora / RHEL

```bash
sudo dnf install -y python3 python3-pip python3-tkinter git-lfs
curl -fsSL https://ollama.com/install.sh | sh
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate
python3 -m venv venv && source venv/bin/activate
pip install -e ".[analysis]"
python3 install.py
```

### Arch / Manjaro

```bash
sudo pacman -S python python-pip tk git-lfs
curl -fsSL https://ollama.com/install.sh | sh
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate
python3 -m venv venv && source venv/bin/activate
pip install -e ".[analysis]"
python3 install.py
```

### Build a standalone binary (optional)

```bash
./build_linux.sh
# → dist/GeneVariate/GeneVariate  (single-directory binary, ~115 MB)
# → GeneVariate.desktop launcher
cp dist/GeneVariate/GeneVariate.desktop ~/.local/share/applications/
```

Continue with [Post-install](#post-install) to pull the models and GEOmetadb.

---

## macOS

### Option 1 — Pre-built `.dmg` (end users)

1. Go to the [Releases page](https://github.com/SciSpectator/genevariate/releases/latest)
2. Download `GeneVariate-x.y.z.dmg`
3. Open the `.dmg` → drag **GeneVariate.app** into **Applications**
4. First launch: right-click → **Open** (unsigned app, Gatekeeper bypass)
5. Install Ollama: `brew install ollama && ollama serve &`

### Option 2 — From source

```bash
# 1. Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. System dependencies
brew install python-tk@3.11 git-lfs ollama

# 3. Start Ollama
ollama serve &

# 4. Clone and install
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate
python3 -m venv venv && source venv/bin/activate
pip install -e ".[analysis]"
python3 install.py
```

### Option 3 — Build your own `.app` + `.dmg`

```bash
./build_mac.sh
# → dist/GeneVariate.app (icon embedded)

# Package into a DMG:
brew install create-dmg
create-dmg --volname "GeneVariate" --window-size 540 380 \
  --icon-size 128 --app-drop-link 380 180 \
  GeneVariate.dmg dist/GeneVariate.app
```

Continue with [Post-install](#post-install).

---

## Windows

### Option 1 — Pre-built installer (end users)

1. Go to the [Releases page](https://github.com/SciSpectator/genevariate/releases/latest)
2. Download `GeneVariate-Setup-x.y.z.exe`
3. Double-click → SmartScreen → **More info** → **Run anyway** (unsigned — source is public)
4. Step through the wizard: license, install location, Start Menu folder, Desktop shortcut
5. Install [Ollama for Windows](https://ollama.com/download/windows)

### Option 2 — From source (PowerShell)

```powershell
# 1. Install Python 3.11 from python.org (tick "Add to PATH")
# 2. Install Git LFS:  https://git-lfs.com

# 3. Clone and install
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate
python -m venv venv
.\venv\Scripts\activate
pip install -e ".[analysis]"

# 4. Desktop shortcut with icon
python install.py

# 5. Ollama
winget install Ollama.Ollama
```

### Option 3 — Build your own `Setup.exe`

```powershell
build_windows.bat
# → dist\GeneVariate\GeneVariate.exe  (icon embedded)
# Use Inno Setup (https://jrsoftware.org/isinfo.php) to produce Setup.exe.
```

Continue with [Post-install](#post-install).

---

## Docker

Docker bundles GeneVariate with its own Ollama server and pulls models automatically.

```bash
# 1. Clone
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate

# 2. Place GEOmetadb
mkdir -p data
# drop GEOmetadb.sqlite.gz into ./data/   (see Post-install for download options)

# 3. Build and launch
docker compose up --build
```

### GUI on Linux (X11)

```bash
xhost +local:docker
docker compose up --build
```

### NVIDIA GPU

Uncomment the `deploy` block under the `ollama` service in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Then `docker compose up --build`.

### Headless mode

```bash
docker compose run genevariate --ns-repair
```

### Volumes

| Host | Container | Purpose |
|---|---|---|
| `./data/` | `/app/src/genevariate/data` | GEOmetadb + platform data |
| `./results/` | `/app/src/genevariate/results` | Analysis output |
| `ollama_models` | `/root/.ollama` | Persistent model storage |

---

## Post-install

After any of the OS paths above, finish with these two steps.

### 1. Pull the Ollama models

```bash
ollama pull gemma4:e2b         # ~2 GB — extraction + ReAct collapse (32k context, unlimited output)
ollama pull nomic-embed-text   # ~274 MB — semantic embeddings + pseudo-cohorts
```

### 2. Get GEOmetadb

GeneVariate needs the NCBI GEO metadata SQLite (`~1.1 GB compressed, ~7 GB decompressed`).
On low-RAM devices it's queried directly from disk with WAL + indexes — no OOM.

**Option A — Git LFS (if you cloned the repo):**
```bash
git lfs install
git lfs pull
# → src/genevariate/data/GEOmetadb.sqlite.gz
```

**Option B — Direct download:**
```bash
# Linux/macOS
wget -O src/genevariate/data/GEOmetadb.sqlite.gz \
  https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz
```

```powershell
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz" `
  -OutFile "src\genevariate\data\GEOmetadb.sqlite.gz"
```

**Option C — R / Bioconductor:**
```r
library(GEOmetadb)
getSQLiteFile(destdir = "src/genevariate/data/")
```

**Updating later** — re-run the same `wget`/`Invoke-WebRequest` to overwrite the `.gz`.

### Launch

```bash
genevariate                # GUI
genevariate --ns-repair    # headless NS-repair pipeline
genevariate-bench --help   # reproducible benchmark harness
```

---

## Troubleshooting

### Ollama not running

```
ConnectionError: Cannot connect to Ollama at http://localhost:11434
```

```bash
ollama serve              # foreground
systemctl start ollama    # systemd
```

### Model not found

```
Error: model "gemma4:e2b" not found
```

```bash
ollama pull gemma4:e2b
ollama pull nomic-embed-text
```

### GEOmetadb not found

```
FileNotFoundError: GEOmetadb.sqlite.gz not found
```

Fetch it via Git LFS (`git lfs pull`) or direct download — see
[Post-install step 2](#2-get-geometadb).

### tkinter not available (Linux)

```
ModuleNotFoundError: No module named 'tkinter'
```

```bash
sudo apt install python3-tk        # Debian/Ubuntu
sudo dnf install python3-tkinter   # Fedora
```

### GPU not detected

```bash
ollama ps         # running models + GPU layers
nvidia-smi        # NVIDIA
rocm-smi          # AMD
```

On Linux, ensure your user is in the right group for GPU access:

```bash
sudo usermod -aG video $USER
# log out and back in
```

### Out of memory

GeneVariate auto-detects your RAM tier and adapts:

| Tier | RAM | GEOmetadb mode | Max workers |
|---|---|---|---|
| Low | ≤ 6 GB | Disk (WAL + mmap) | 4 |
| Medium | 6–14 GB | Disk or RAM | 20 |
| High | ≥ 14 GB | Full in-memory | 210 |

If OOM still happens: close other apps (especially browsers), keep the default
`gemma4:e2b` model (already memory-efficient), let the watchdog hard-pause under pressure.

### Optional analysis dependencies missing

The novel-enrichment methods need the `analysis` extra:

```bash
pip install -e ".[analysis]"
# or for ARCHS4 bulk RNA-seq + DESeq2:
pip install -e ".[analysis,rnaseq]"
```

---

## Project layout

```
genevariate/
├── docs/
│   ├── logo.png                          # Official logo
│   └── architecture.svg                  # Pipeline diagram
├── src/genevariate/
│   ├── main.py                           # Entry point
│   ├── config.py                         # Config + GPU auto-detection
│   ├── assets/
│   │   └── icon.png                      # App icon
│   ├── core/
│   │   ├── ai_engine.py                  # 8-class distribution classifier
│   │   ├── db_loader.py                  # Shared GEOmetadb loader
│   │   ├── extraction.py                 # LLM prompts + parsers + Phase 1.5 rules
│   │   ├── gpl_downloader.py             # GPL annotations, probe-to-gene, qnorm
│   │   ├── gse_worker.py                 # Per-GSE extraction agent
│   │   ├── gse_context.py                # Rolling per-experiment context
│   │   ├── memory_agent.py               # 4-tier persistent memory
│   │   ├── ns_repair_pipeline.py         # Multi-phase NS repair orchestrator
│   │   ├── ollama_manager.py             # Watchdog + thermal guard + GPU detection
│   │   ├── statistics.py                 # Wilcoxon / Welch / Wasserstein / Cohen
│   │   ├── nlp.py                        # Rule-based sample classification
│   │   ├── sources/
│   │   │   ├── base.py                   # Canonical format contract
│   │   │   └── archs4.py                 # ARCHS4 bulk RNA-seq ingestion
│   │   └── analysis/
│   │       ├── variability.py            # ΔVariance ranking + GSEA
│   │       ├── enrichment.py             # Mean-based Enrichr / GSEA
│   │       ├── meta_enrichment.py        # Rank-product / Stouffer combination
│   │       ├── bimodality.py             # Distribution-gated gene filtering
│   │       └── pseudo_cohorts.py         # Embedding-clustered auto-cohorts
│   ├── gui/                              # Tkinter GUI
│   ├── utils/                            # Worker threads, plotting
│   ├── memory/                           # Persistent memory storage
│   └── data/                             # GEOmetadb.sqlite.gz (Git LFS)
├── tests/                                # pytest suite (core + analysis + sources)
├── pyproject.toml                        # Package metadata and extras
├── Dockerfile, docker-compose.yml        # Container build
├── build_linux.sh, build_mac.sh, build_windows.bat
├── genevariate.spec                      # PyInstaller spec
├── LICENSE                               # MIT
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).
