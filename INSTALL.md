# Installation

Pick the section that matches your setup.

- [Linux](#linux)
- [macOS](#macos)
- [Windows](#windows)
- [Docker](#docker)
- [After install](#after-install)
- [Optional extras](#optional-extras)
- [Language model endpoint](#language-model-endpoint)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

Every path below ends at the same place: `genevariate` running on your machine.

The program needs no language model to run. Loading platforms, brushing regions
and every analysis in the region window work with no endpoint configured. Only
the label extractor and the assistant need one.

## Linux

### Ubuntu or Debian

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-tk git-lfs

git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis]"

genevariate
```

`python3 install.py` adds a desktop launcher if you want one.

### Fedora or RHEL

```bash
sudo dnf install -y python3 python3-pip python3-tkinter git-lfs
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]"
```

### Arch or Manjaro

```bash
sudo pacman -S python python-pip tk git-lfs
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]"
```

### Standalone binary

```bash
./build_linux.sh
cp dist/GeneVariate/GeneVariate.desktop ~/.local/share/applications/
```

Produces `dist/GeneVariate/GeneVariate`, a single-directory binary.

## macOS

### From source

```bash
brew install python-tk@3.11 git-lfs

git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]"

genevariate
```

Homebrew itself, if you do not have it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Build an app bundle

```bash
./build_mac.sh

brew install create-dmg
create-dmg --volname "GeneVariate" --window-size 540 380 \
  --icon-size 128 --app-drop-link 380 180 \
  GeneVariate.dmg dist/GeneVariate.app
```

The bundle is unsigned, so the first launch needs right-click then **Open**.

## Windows

Install Python 3.11 from python.org with **Add to PATH** ticked, and
[Git LFS](https://git-lfs.com).

```powershell
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[analysis]"

genevariate
```

`python install.py` adds a desktop shortcut with the app icon.

### Build an installer

```powershell
build_windows.bat
```

Produces `dist\GeneVariate\GeneVariate.exe`. Use
[Inno Setup](https://jrsoftware.org/isinfo.php) to wrap it into a `Setup.exe`.

## Docker

```bash
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate

mkdir -p data results
# put GEOmetadb.sqlite.gz into ./data/ (see "After install")

docker compose build
docker compose run --rm genevariate
```

No model weights are baked into the image. If you want the extractor or the
assistant, run an inference server on the host and the container will reach it
through `host.docker.internal`.

### GUI on Linux

```bash
xhost +local:docker
docker compose run --rm genevariate
```

### Volumes

| Host | Container | Purpose |
|---|---|---|
| `./data/` | `/app/src/genevariate/data` | GEOmetadb and platform data |
| `./results/` | `/app/src/genevariate/results` | analysis output |

## After install

### GEOmetadb

GeneVariate uses the NCBI GEO metadata SQLite catalogue to resolve a sample to
the study it came from, which is what every study-clumping correction depends
on. It is about 1.1 GB compressed and 7 GB decompressed, and on low-RAM machines
it is queried directly from disk with WAL and indexes rather than being loaded.

The database is **not** in the repository and is not fetched by cloning; it is
too large to distribute and is rebuilt upstream. Download it:

```bash
wget -O src/genevariate/data/GEOmetadb.sqlite.gz \
  https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz
```

```powershell
Invoke-WebRequest -Uri "https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz" `
  -OutFile "src\genevariate\data\GEOmetadb.sqlite.gz"
```

Through Bioconductor:

```r
library(GEOmetadb)
getSQLiteFile(destdir = "src/genevariate/data/")
```

Re-run the same download to update it later.

An expression CSV that already carries a `series_id` column skips the lookup
entirely, so a custom platform works without GEOmetadb.

### Controlled vocabularies (label extraction only)

The label extractor resolves Tissue, Condition and Treatment against MeSH and
Cellosaurus. Those databases are **not in the repository** and are not
downloaded automatically; without them the extractor still runs but stops after
the context-inference pass and emits no accession for any field. Everything
else in the program, including loading a platform and attaching a label file
you already have, works without them.

Four artifacts are looked for, in `refs/` beside the vendored extractor or in
`rnaseq_labels/refs/` at the checkout root, and each can be pointed elsewhere
with an environment variable:

| File | Variable | Built from |
|---|---|---|
| `mesh.sqlite` | `MESH_DB` | NLM MeSH descriptors, `desc<year>.xml` |
| `vocab.sqlite` | `GEO_VOCAB` | the same descriptors, as the lookup vocabulary |
| `vocab_index.npz` | `GEO_INDEX` | sentence-transformer embeddings of that vocabulary (about 1.7 GB) |
| `cellosaurus.sqlite` | `GEO_CELLOSAURUS` | `cellosaurus.txt` from the Cellosaurus release |

Download the two sources, then build:

The builders read their input and write their output from environment
variables, so run them from the directory that will hold the artifacts:

```bash
mkdir -p rnaseq_labels/refs && cd rnaseq_labels/refs
wget https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml
wget https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt

E=../../src/genevariate/_vendor/geo_label_extractor
MESH_XML=desc2026.xml       MESH_DB=mesh.sqlite            python3 $E/build_mesh_db.py
CELLOSAURUS_TXT=cellosaurus.txt CELLLINE_DB=cellosaurus.sqlite python3 $E/build_cellosaurus_db.py
python3 $E/build_vocab_from_meshdb.py mesh.sqlite vocab.sqlite
python3 $E/build_index.py vocab.sqlite cellosaurus.sqlite vocab_index.npz
```

The last step embeds the vocabulary and needs `sentence-transformers` (the
`llm-extract` extra); it is the slow one and produces the 1.7 GB file. Budget
about 2.5 GB of disk for all four artifacts.

### Launch

```bash
genevariate                    # the program
genevariate-llm-extract --help # the vendored label extractor, headless
```

### Check the install

```bash
python3 tools/make_synthetic_platform.py
python3 tools/validate_synthetic.py
```

24 claims are checked against a ground truth written before the analysis ran.
All 24 should pass.

## Optional extras

The base install runs the program. Each extra adds a feature and nothing else,
and when one is missing the feature that needs it raises a clear error while the
rest of the program keeps working.

```bash
pip install -e ".[analysis]"     # gseapy, archs4py, mygene, statsmodels
pip install -e ".[bimodality]"   # diptest, for the Hartigan dip gate
pip install -e ".[activity]"     # decoupler, for TF and pathway activity
pip install -e ".[integration]"  # inmoose and harmonypy, for ComBat and Harmony
pip install -e ".[llm-extract]"  # dspy, sentence-transformers, torch
pip install -e ".[agent]"        # langchain, for the assistant
pip install -e ".[all]"          # everything above
```

## Language model endpoint

The label extractor and the assistant talk to an OpenAI-compatible endpoint over
HTTP. Any local server such as vLLM, SGLang or Ollama will serve them, and so
will a hosted one if you point them at it.

```bash
export GENEVARIATE_AGENT_BACKEND=ollama            # or groq, or openai-compatible
export GENEVARIATE_LLM_URL=http://127.0.0.1:11434/v1
export GENEVARIATE_AGENT_MODEL=<model name>
```

No model runs inside the program and no weights ship with it. Nothing is sent
anywhere you did not configure.

## Troubleshooting

### tkinter not available

```text
ModuleNotFoundError: No module named 'tkinter'
```

```bash
sudo apt install python3-tk        # Debian or Ubuntu
sudo dnf install python3-tkinter   # Fedora
brew install python-tk@3.11        # macOS
```

### GEOmetadb not found

```text
FileNotFoundError: GEOmetadb.sqlite.gz not found
```

Download it as shown in
[After install](#after-install). A custom CSV that carries `series_id` does not
need it.

### The assistant cannot reach a model

```text
ConnectionError: cannot reach the configured endpoint
```

Start your inference server and check `GENEVARIATE_LLM_URL`. Everything except
the assistant and the extractor works without one.

### An analysis says a package is missing

Install the extra that carries it, listed under
[Optional extras](#optional-extras). Nothing else in the program is affected.

### Memory pressure

GeneVariate detects the available RAM and adapts how GEOmetadb is opened and how
many workers it runs.

| Tier | RAM | GEOmetadb | Workers |
|---|---|---|---|
| Low | up to 6 GB | disk, WAL and mmap | 4 |
| Medium | 6 to 14 GB | disk or memory | 20 |
| High | 14 GB and above | fully in memory | 210 |

If a machine still runs out, close other applications and load fewer platforms
at once. A platform frame is held in memory for as long as it is loaded.

## Project layout

```text
genevariate/
├── docs/
│   ├── architecture.md, methods.md, tutorial.md
│   ├── output-schema.md, reproducibility.md
│   ├── architecture.svg
│   └── logo.png
├── src/genevariate/
│   ├── main.py                  entry point
│   ├── config.py                configuration and resource tiers
│   ├── core/
│   │   ├── gpl_downloader.py    GPL annotations, probe to gene, normalization
│   │   ├── rnaseq_counts.py     count matrices and TMM factors
│   │   ├── db_loader.py         GEOmetadb and custom CSV loading
│   │   ├── label_entities.py    MeSH, Cellosaurus and local namespaces
│   │   ├── geo_extract_driver.py  drives the vendored label extractor
│   │   ├── llm_client.py        OpenAI-compatible HTTP chat
│   │   ├── reproducibility.py   per-run manifest
│   │   ├── external_enrichment.py
│   │   ├── memory_agent.py, ollama_manager.py
│   │   ├── sources/             canonical format contract, ARCHS4
│   │   ├── chatbot/             the assistant, tools and backends
│   │   └── analysis/
│   │       ├── overdispersion.py    rho, design effect, effective n
│   │       ├── pooled_enrichment.py DerSimonian-Laird across platforms
│   │       ├── region_comparison.py regions against each other
│   │       ├── synergy.py           k-way interaction for conjunction boxes
│   │       ├── box_model.py         calibrated P(label given expression)
│   │       ├── enrichment.py, meta_enrichment.py
│   │       ├── variability.py, bimodality.py
│   │       ├── integration.py, cross_modality.py
│   │       ├── activity.py, label_ml.py, pseudo_cohorts.py
│   ├── sources/                 ARCHS4, CELLxGENE, experiment discovery
│   ├── gui/                     app, region and cross-platform windows, theme
│   ├── utils/                   worker threads, pseudobulk, plotting, export
│   └── data/                    GEOmetadb.sqlite.gz (downloaded, not in git)
├── tools/                       synthetic platform generator and validator
├── pyproject.toml
├── Dockerfile, docker-compose.yml
├── build_linux.sh, build_mac.sh, build_windows.bat
├── genevariate.spec
├── LICENSE
└── README.md
```

## License

MIT. See [LICENSE](LICENSE).
