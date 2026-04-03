FROM python:3.11-slim

# System dependencies (tkinter for GUI, X11 libs for display forwarding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxft2 \
    libxss1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY pyproject.toml .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir -e .

# Create data directory (user mounts their data here)
RUN mkdir -p src/genevariate/data src/genevariate/results src/genevariate/cache

# Default: launch the GUI
# Override with --ns-repair for headless pipeline mode
ENTRYPOINT ["genevariate"]
