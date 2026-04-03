# GeneVariate v2.0 — Installation Guide

Complete setup instructions for running GeneVariate on **any local device** (GPU or CPU-only).

---

## Quick Start (5 minutes)

```bash
# 1. Clone (with Git LFS for GEOmetadb)
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows

# 3. Install
pip install -e .

# 4. Install Ollama (LLM engine)
curl -fsSL https://ollama.com/install.sh | sh   # Linux/macOS
# Windows: download from https://ollama.com/download

# 5. Pull required models
ollama pull gemma2:2b
ollama pull gemma2:9b
ollama pull nomic-embed-text

# 6. Download GEOmetadb (if not pulled via Git LFS)
#    Skip this if `git lfs pull` already fetched it
wget -O src/genevariate/data/GEOmetadb.sqlite.gz \
  https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz

# 7. Launch
genevariate
```

---

## Detailed Installation

### Step 1: System Requirements

| Resource | Minimum (low-RAM mode) | Recommended |
|----------|---------|-------------|
| **OS** | Linux, macOS, Windows | Ubuntu 22.04+ / macOS 13+ |
| **Python** | 3.9+ | 3.11+ |
| **RAM** | 4 GB | 16+ GB |
| **Storage** | 3 GB (code + models + GEOmetadb) | 10+ GB |
| **GPU** | None (CPU works) | NVIDIA 6+ GB VRAM |

> **Low-RAM devices (4-6 GB):** GeneVariate automatically uses disk-based GEOmetadb queries, smaller batch sizes, and fewer workers. All features work -- just slower.

### Step 2: Install System Dependencies

#### Tkinter (required for GUI)

Tkinter is part of Python's standard library but may need a system package:

```bash
# Ubuntu / Debian
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter

# macOS (Homebrew Python)
brew install python-tk@3.11

# Windows
# Included with official Python installer (check "tcl/tk" during install)
```

#### Ollama (required for AI features)

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from: https://ollama.com/download

# Verify installation
ollama --version
```

### Step 3: Install GeneVariate

```bash
# Unzip the archive
unzip GeneVariate_v2.0.zip
cd genevariate

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package (auto-installs all Python dependencies)
pip install -e .
```

**Dependencies installed automatically:**
numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, GEOparse, requests, psutil, ollama, qnorm

### Step 4: Pull Ollama Models

GeneVariate uses three local LLM models:

```bash
# Start Ollama server (if not already running)
ollama serve &

# Pull models (one-time download)
ollama pull gemma2:2b          # ~1.5 GB — fast extraction
ollama pull gemma2:9b          # ~5.4 GB — collapse reasoning
ollama pull nomic-embed-text   # ~274 MB — semantic embeddings
```

### Step 5: Download GEOmetadb

GEOmetadb is the SQLite database of all NCBI GEO experiment metadata. GeneVariate requires it for searching and retrieving experiment information.

**Option A — Git LFS (if you cloned the repo):**
```bash
git lfs install
git lfs pull
# GEOmetadb is now at: src/genevariate/data/GEOmetadb.sqlite.gz
```

**Option B — Direct download:**
```bash
# Linux / macOS
wget -O src/genevariate/data/GEOmetadb.sqlite.gz \
  https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz

# Or with curl
curl -L -o src/genevariate/data/GEOmetadb.sqlite.gz \
  https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz
```

Windows (PowerShell):
```powershell
Invoke-WebRequest -Uri "https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz" `
  -OutFile "src\genevariate\data\GEOmetadb.sqlite.gz"
```

**Option C — From R (Bioconductor):**
```r
library(GEOmetadb)
getSQLiteFile(destdir = "src/genevariate/data/")
```

> **Size:** ~1.1 GB compressed, ~7 GB decompressed. On low-RAM devices, GeneVariate queries it directly from disk (WAL mode + indexes) without loading into memory.

**To update** (get latest experiments):
```bash
wget -O src/genevariate/data/GEOmetadb.sqlite.gz \
  https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz
```

### Step 6: Launch

```bash
# GUI mode (default)
genevariate

# Or run directly
python -m genevariate.main

# NS Repair pipeline (headless)
genevariate --ns-repair
```

---

## Alternative: Docker Installation

Docker bundles everything (Ollama + models + GeneVariate) in one command.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Run

```bash
cd genevariate

# Create data directory and add GEOmetadb
mkdir -p data
# Place GEOmetadb.sqlite.gz in ./data/

# Build and launch (pulls models automatically)
docker compose up --build
```

### GUI mode (Linux with X11)

```bash
xhost +local:docker
docker compose up --build
```

### GPU support (NVIDIA)

Edit `docker-compose.yml` — uncomment the `deploy` section under the `ollama` service:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Then: `docker compose up --build`

### Headless mode

```bash
docker compose run genevariate --ns-repair
```

---

## Hardware Adaptation

GeneVariate automatically detects your hardware at startup and classifies your device into a resource tier:

| Tier | RAM | GEOmetadb | Max Workers | Batch Size |
|------|-----|-----------|-------------|------------|
| **Low** | <= 6 GB | Disk (WAL + indexes) | 4 | 50 |
| **Medium** | 6-14 GB | Disk or RAM | 20 | 100 |
| **High** | >= 14 GB | Full in-memory | 210 | 200 |

### GPU Detection

- **NVIDIA**: detected via `nvidia-smi` (CUDA)
- **AMD**: detected via `rocm-smi` (ROCm)
- **No GPU**: falls back to CPU-only mode automatically

### Fluid Agent Scaling

The resource watchdog dynamically adjusts concurrency with **tier-adapted thresholds**:

| Condition | Low-RAM | High-RAM |
|-----------|---------|----------|
| RAM above high threshold | Scale down (80%) | Scale down (92%) |
| RAM below low threshold | Scale up (+1) | Scale up (+20) |
| Near OOM | Hard pause (92%) | Hard pause (99%) |
| Thermal limit | Hard pause | Hard pause |
| Recovery | Auto-resume | Auto-resume |

### CPU-Only / Low-RAM Operation

On machines without a GPU or with limited RAM:
- All features work -- inference runs on CPU (slower but functional)
- Worker count is auto-calculated based on available RAM
- GEOmetadb is queried from disk with SQLite WAL mode, indexes, and mmap
- The `gemma2:2b` model runs on 4 GB RAM with CPU

---

## Troubleshooting

### "Ollama not found"
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
# Then start it
ollama serve
```

### "tkinter not found"
```bash
# Ubuntu/Debian
sudo apt install python3-tk
```

### "GEOmetadb not found"
The app will warn you at startup. Download it:
```bash
wget -O src/genevariate/data/GEOmetadb.sqlite.gz \
  https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz
```

### Models not pulling
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags
# If not, start it:
ollama serve &
# Then pull:
ollama pull gemma2:2b
```

### Out of Memory (OOM)
- GeneVariate auto-detects your RAM and adapts (4 GB devices use disk-based DB, fewer workers)
- The watchdog automatically scales down workers on memory pressure
- On very low-RAM systems (< 4 GB), use `gemma2:2b` only by editing `config.py`:
  ```python
  'model': 'gemma2:2b',
  ```

### Permission errors on Linux
```bash
# If Ollama can't access GPU:
sudo usermod -aG video $USER
# Then log out and back in
```

---

## Project Structure

```
genevariate/
├── src/genevariate/           # Source code
│   ├── main.py                # Entry point
│   ├── config.py              # Configuration (auto-detect GPU/CPU)
│   ├── core/                  # AI engine, extraction, memory, statistics
│   ├── gui/                   # Tkinter GUI application
│   ├── utils/                 # Worker threads, plotting
│   └── memory/                # Persistent memory storage
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Package metadata
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Docker orchestration
├── LICENSE                    # MIT License
├── README.md                  # Project documentation
└── INSTALL.md                 # This file
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
