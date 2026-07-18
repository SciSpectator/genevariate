"""
GeneVariate — pluggable data-source layer.

Each module here provides a thin function / class that fetches real data
from a public bioinformatics resource and returns it either as a
GeneVariate platform DataFrame or an AnnData (convertible to the same
shape via ``utils.anndata_io.anndata_to_platform_df``).

Currently implemented:
    * ``cellxgene``  — CELLxGENE Discover Census (single-cell, ~50M cells)

Planned:
    * ``archs4``     — ARCHS4 (uniformly processed RNA-seq from GEO/SRA)
    * ``refinebio``  — refine.bio (cross-platform normalized bulk)
    * ``xena``       — UCSC Xena (TCGA + GTEx + CCLE)
"""
