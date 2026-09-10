"""
GeneVariate Memory System

Persistent memory storage for LLM extraction agents.
All data stored in SQLite databases by default.

Directory layout:
    memory/
    ├── clusters/      # Cluster vocabulary DB (biomedical_memory.db)
    ├── episodic/      # Episodic resolution logs
    ├── context/       # Context window snapshots per GSE
    └── embeddings/    # Cached vector embeddings
"""

import os

MEMORY_ROOT = os.path.dirname(os.path.abspath(__file__))
CLUSTERS_DIR = os.path.join(MEMORY_ROOT, "clusters")
EPISODIC_DIR = os.path.join(MEMORY_ROOT, "episodic")
CONTEXT_DIR = os.path.join(MEMORY_ROOT, "context")
EMBEDDINGS_DIR = os.path.join(MEMORY_ROOT, "embeddings")

# Ensure directories exist
for d in (CLUSTERS_DIR, EPISODIC_DIR, CONTEXT_DIR, EMBEDDINGS_DIR):
    os.makedirs(d, exist_ok=True)
