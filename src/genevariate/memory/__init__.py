"""GeneVariate Memory System"""

import os

MEMORY_ROOT = os.path.dirname(os.path.abspath(__file__))
CLUSTERS_DIR = os.path.join(MEMORY_ROOT, "clusters")
EPISODIC_DIR = os.path.join(MEMORY_ROOT, "episodic")
CONTEXT_DIR = os.path.join(MEMORY_ROOT, "context")
EMBEDDINGS_DIR = os.path.join(MEMORY_ROOT, "embeddings")

# Ensure directories exist
for d in (CLUSTERS_DIR, EPISODIC_DIR, CONTEXT_DIR, EMBEDDINGS_DIR):
    os.makedirs(d, exist_ok=True)
