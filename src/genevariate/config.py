"""
GeneVariate Configuration
Central configuration for all paths, settings, and parameters.
Resource-aware: automatically adapts to available RAM and hardware.
"""

import os
from pathlib import Path

import psutil

BASE_DIR = Path(__file__).parent


# ── Resource-aware device classification ──────────────────────────────────
# Classifies the machine so every subsystem (DB loading, worker counts,
# batch sizes) can adapt without the user touching config.

def _detect_resource_tier():
    """Classify the device based on total RAM.

    Returns a dict with tier name and adaptive defaults:
      - 'low'    : <= 6 GB RAM  (e.g. old laptop, Raspberry Pi, cheap VPS)
      - 'medium' : 6-14 GB RAM
      - 'high'   : >= 14 GB RAM
    """
    total_gb = psutil.virtual_memory().total / (1024 ** 3)

    if total_gb <= 6:
        return {
            'tier': 'low',
            'total_ram_gb': round(total_gb, 1),
            'db_in_memory': False,       # use disk-based SQLite
            'max_workers': 2,
            'batch_size': 50,
            'checkpoint_every': 200,
            'ncbi_workers': 2,
            'watchdog_min_workers': 1,
            'watchdog_max_workers': 4,
            'watchdog_scale_up_step': 1,
            'ram_high_pct': 80.0,
            'ram_low_pct': 60.0,
            'ram_pause_pct': 92.0,
        }
    elif total_gb <= 14:
        return {
            'tier': 'medium',
            'total_ram_gb': round(total_gb, 1),
            'db_in_memory': total_gb >= 10,  # only in-memory if >= 10 GB
            'max_workers': 4,
            'batch_size': 100,
            'checkpoint_every': 500,
            'ncbi_workers': 3,
            'watchdog_min_workers': 2,
            'watchdog_max_workers': 20,
            'watchdog_scale_up_step': 4,
            'ram_high_pct': 85.0,
            'ram_low_pct': 70.0,
            'ram_pause_pct': 95.0,
        }
    else:
        return {
            'tier': 'high',
            'total_ram_gb': round(total_gb, 1),
            'db_in_memory': True,
            'max_workers': 4,
            'batch_size': 200,
            'checkpoint_every': 1000,
            'ncbi_workers': 5,
            'watchdog_min_workers': 4,
            'watchdog_max_workers': 210,
            'watchdog_scale_up_step': 20,
            'ram_high_pct': 92.0,
            'ram_low_pct': 80.0,
            'ram_pause_pct': 99.0,
        }


RESOURCE_TIER = _detect_resource_tier()


CONFIG = {
    'paths': {
        'data': BASE_DIR / 'data',
        'results': BASE_DIR / 'results',
        'cache': BASE_DIR / 'cache',
        'geo_db': BASE_DIR / 'data' / 'GEOmetadb.sqlite.gz',
        'gpl570_data': BASE_DIR / 'data' / 'GPL570_data.csv.gz',
        'gpl570_meta': BASE_DIR / 'data' / 'GPL570_metadata.csv.gz',
        'gpl96_data': BASE_DIR / 'data' / 'GPL96_data.csv.gz',
        'gpl6947_data': BASE_DIR / 'data' / 'GPL6947_data.csv.gz',
        'gpl10558_data': BASE_DIR / 'data' / 'GPL10558_data.csv.gz',
        'gpl6885_data': BASE_DIR / 'data' / 'GPL6885_data.csv.gz',
        'gpl1261_data': BASE_DIR / 'data' / 'GPL1261_data.csv.gz',
        'gpl7202_data': BASE_DIR / 'data' / 'GPL7202_data.csv.gz',
        'embedding_cache': BASE_DIR / 'cache' / 'embeddings',
        'model_cache': BASE_DIR / 'cache' / 'models',
    },
    'database': {
        'sql_chunk_size': 500,
        'connection_timeout': 30,
        'max_retries': 3,
        'in_memory': RESOURCE_TIER['db_in_memory'],
    },
    'threading': {
        'max_workers': RESOURCE_TIER['max_workers'],
        'timeout': 300,
    },
    'ai': {
        'model': 'gemma4:e2b',
        'extraction_model': 'gemma4:e2b',
        'embedding_model': 'nomic-embed-text',
        'device': 'auto',  # auto-detect GPU/CPU at runtime
        'temperature': 0,
        'max_tokens': 60,
        'num_ctx': 512,
        'think': False,             # disable gemma4 reasoning chain for speed
        'timeout': 30,
        'ollama_url': os.environ.get('OLLAMA_HOST', 'http://localhost:11434'),
    },
    'memory': {
        'db_name': 'biomedical_memory.db',
        'memory_root': BASE_DIR / 'memory',
        'clusters_dir': BASE_DIR / 'memory' / 'clusters',
        'episodic_dir': BASE_DIR / 'memory' / 'episodic',
        'context_dir': BASE_DIR / 'memory' / 'context',
        'embeddings_dir': BASE_DIR / 'memory' / 'embeddings',
        'embed_batch_size': 32,
        'semantic_top_k': 10,
        'match_threshold': 0.72,
    },
    'extraction': {
        'label_cols': ['Tissue', 'Condition'],
        'label_cols_scratch': ['Tissue', 'Condition', 'Treatment'],
        'batch_size': RESOURCE_TIER['batch_size'],
        'checkpoint_every': RESOURCE_TIER['checkpoint_every'],
        'ncbi_workers': RESOURCE_TIER['ncbi_workers'],
        'ncbi_delay': 0.35,
        'platforms': ['GPL6947', 'GPL96', 'GPL570', 'GPL10558'],
    },
    'plotting': {
        'figure': {
            'dpi': 100,
            'subplot_size': (5, 4),
            'title_fontsize': 12,
            'label_fontsize': 10,
            'tick_fontsize': 8,
        },
        'histogram': {
            'default_color': '#3498db',
            'edge_color': '#2c3e50',
            'alpha': 0.7,
            'min_bins': 10,
            'max_bins': 100,
            'min_samples_for_kde': 30,
            'min_variance_for_kde': 0.01,
        },
        'selection': {
            'face_color': 'yellow',
            'edge_color': 'red',
            'alpha': 0.3,
            'linewidth': 2,
        },
        'pca': {
            'point_size': 50,
            'alpha': 0.6,
            'n_components': 2,
        },
        'dpc': {
            'point_size': 50,
            'alpha': 0.7,
            'dc_percentile': 2,
        },
        'colors': {
            'categorical': [
                '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#16a085',
                '#27ae60', '#2980b9', '#8e44ad', '#f1c40f', '#d35400',
            ],
            'sequential': 'viridis',
            'diverging': 'RdBu',
        },
    },
    'statistics': {
        'alpha': 0.05,
        'min_samples_per_group': 3,
        'outlier_threshold': 3.0,
    },
    'export': {
        'dpi': 300,
        'format': 'png',
        'bbox_inches': 'tight',
    },
    'ui': {
        'window_geometry': '1100x980',
        'log_max_lines': 1000,
        'theme': 'clam',
    },
}

METADATA_EXCLUSIONS = {
    'GSM', 'gsm', 'ID', 'id', 'Sample', 'sample', 'SampleID', 'sample_id',
    'Series', 'series', 'series_id', 'GPL', 'gpl', 'Platform', 'platform',
    'Title', 'title', 'Description', 'description', 'Source', 'source',
    'Organism', 'organism', 'Characteristics', 'characteristics',
    'Treatment', 'treatment', 'Protocol', 'protocol', 'Extract', 'extract',
    'Label', 'label', 'Hybridization', 'hybridization', 'Scan', 'scan',
    'Data_Processing', 'data_processing', 'Value', 'value',
    'Submission_Date', 'submission_date', 'Last_Update_Date', 'last_update_date',
    'Type', 'type', 'Channel_Count', 'channel_count', 'Source_Name', 'source_name',
    'Molecule', 'molecule', 'Extract_Protocol', 'extract_protocol',
    'Label_Protocol', 'label_protocol', 'Hyb_Protocol', 'hyb_protocol',
    'Scan_Protocol', 'scan_protocol', 'Description_1', 'Description_2',
    'Characteristics_1', 'Characteristics_2', 'Characteristics_3',
    'Contact', 'contact', 'Supplementary_File', 'supplementary_file',
    'Data_Row_Count', 'data_row_count', 'Status', 'status',
    'Taxid', 'taxid', 'Relation', 'relation',
    'SAMPLE_ID', 'SERIES_ID', 'PLATFORM', 'TITLE', 'DESCRIPTION',
    'sample name', 'Sample Name', 'Sample_Name', 'SAMPLE_NAME',
    'geo_accession', 'GEO_Accession', 'GEO Accession',
    'Classified_Condition', 'Classified_Tissue', 'Classified_Treatment',
    'classified_condition', 'classified_tissue', 'classified_treatment',
    'CLASSIFIED_CONDITION', 'CLASSIFIED_TISSUE', 'CLASSIFIED_TREATMENT',
    'Cluster', 'cluster', 'Group', 'group', 'Category', 'category',
    'Class', 'class', 'Region', 'region', 'Selection', 'selection',
    'Gene Expression Value', 'gene_expression_value', 'expression_value',
}

DISTRIBUTION_PATTERNS = {
    'normal': {
        'skewness_range': (-0.5, 0.5),
        'kurtosis_range': (-1.0, 1.0),
    },
    'lognormal': {
        'skewness_min': 0.5,
        'positive_only': True,
    },
    'bimodal': {
        'min_peaks': 2,
        'peak_prominence': 0.1,
    },
    'uniform': {
        'kurtosis_max': -1.0,
    },
}

GOLDEN_RATIO = 1.618033988749895

def get_selection_colors(n=25):
    """Generate n distinct colors using golden ratio spacing."""
    import colorsys
    colors = []
    for i in range(n):
        hue = (i * GOLDEN_RATIO) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


def init_directories():
    """Create required directories if they don't exist."""
    for path_name, path_value in CONFIG['paths'].items():
        if path_name.endswith('_cache') or path_name in ['data', 'results', 'cache']:
            path_value.mkdir(parents=True, exist_ok=True)


def validate_config():
    """Validate configuration settings."""
    errors = []
    
    if not CONFIG['paths']['data'].exists():
        errors.append(f"Data directory not found: {CONFIG['paths']['data']}")
    
    if CONFIG['ai']['temperature'] < 0 or CONFIG['ai']['temperature'] > 1:
        errors.append("AI temperature must be between 0 and 1")
    
    if CONFIG['ai']['max_tokens'] < 1:
        errors.append("AI max_tokens must be positive")
    
    if CONFIG['threading']['max_workers'] < 1:
        errors.append("max_workers must be at least 1")
    
    if CONFIG['plotting']['histogram']['min_bins'] < 5:
        errors.append("min_bins must be at least 5")
    
    if CONFIG['plotting']['histogram']['max_bins'] < CONFIG['plotting']['histogram']['min_bins']:
        errors.append("max_bins must be >= min_bins")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  • {e}" for e in errors))
    
    return True


init_directories()

try:
    validate_config()
except ValueError as e:
    print(f"⚠️  Configuration Warning: {e}")

def detect_device() -> str:
    """Return 'gpu' if any NVIDIA/AMD GPU is available, else 'cpu'.
    Called lazily so config.py import never crashes on headless/minimal systems.
    """
    import subprocess
    try:
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=5)
        return "gpu"
    except Exception:
        pass
    try:
        subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            stderr=subprocess.DEVNULL, text=True, timeout=5)
        return "gpu"
    except Exception:
        pass
    return "cpu"


def get_device() -> str:
    """Resolve the configured device. 'auto' triggers runtime GPU detection."""
    dev = CONFIG['ai']['device']
    if dev == 'auto':
        return detect_device()
    return dev


__all__ = [
    'CONFIG',
    'RESOURCE_TIER',
    'METADATA_EXCLUSIONS',
    'DISTRIBUTION_PATTERNS',
    'GOLDEN_RATIO',
    'get_selection_colors',
    'init_directories',
    'validate_config',
    'detect_device',
    'get_device',
]
