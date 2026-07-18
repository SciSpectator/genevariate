"""
GeneVariate — Centralized plot export manager.

Gives every plot window a single, consistent way to save figures:
* Auto-named files:  ``{analysis_id}_{plot_key}_{timestamp}.{ext}``
* 300 DPI + ``bbox_inches='tight'`` everywhere
* Batch "save all" for the whole window
* Optional HTML gallery index so users can browse a day's exports
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import re as _re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

from .viz_style import EXPORT_DPI


_SAFE_CHARS = _re.compile(r"[^A-Za-z0-9._-]+")


def _slug(name: str) -> str:
    s = _SAFE_CHARS.sub("_", str(name or "plot")).strip("_")
    return s or "plot"


class PlotExportManager:
    """Batch-friendly matplotlib export helper.

    Usage
    -----
    >>> mgr = PlotExportManager(base_dir=Path.home() / 'genevariate_exports')
    >>> mgr.export_figure(fig, plot_key='pca', analysis_id='GSE1234')
    >>> mgr.export_batch({'pca': fig1, 'umap': fig2}, analysis_id='GSE1234')
    """

    def __init__(self, base_dir: Optional[Path] = None,
                 dpi: int = EXPORT_DPI,
                 default_format: str = "png"):
        self.base_dir = Path(base_dir) if base_dir else \
            (Path.home() / "genevariate_exports")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = int(dpi)
        self.default_format = default_format.lower().lstrip(".")

    # ------------------------------------------------------------------
    def auto_filename(self, plot_key: str, analysis_id: str = "analysis",
                      ext: Optional[str] = None,
                      timestamp: bool = True) -> str:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""
        pieces = [_slug(analysis_id), _slug(plot_key)]
        if ts:
            pieces.append(ts)
        ext = (ext or self.default_format).lower().lstrip(".")
        return f"{'_'.join(pieces)}.{ext}"

    # ------------------------------------------------------------------
    def export_figure(self, fig, plot_key: str,
                      analysis_id: str = "analysis",
                      ext: Optional[str] = None,
                      subdir: Optional[str] = None,
                      metadata: Optional[Mapping[str, str]] = None) -> Path:
        """Save one figure; return the destination path."""
        fn = self.auto_filename(plot_key, analysis_id, ext=ext)
        out_dir = (self.base_dir / _slug(subdir)) if subdir else self.base_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / fn

        kwargs = dict(dpi=self.dpi, bbox_inches="tight")
        fmt = (ext or self.default_format).lower().lstrip(".")
        if fmt in ("pdf", "svg"):
            kwargs["metadata"] = {
                "Creator": "GeneVariate",
                "Title": f"{analysis_id} — {plot_key}",
                "Subject": "Gene expression analysis plot",
            }
            if metadata:
                kwargs["metadata"].update({str(k): str(v)
                                            for k, v in metadata.items()})
        fig.savefig(path, **kwargs)
        return path

    # ------------------------------------------------------------------
    def export_batch(self, figures: Mapping[str, object],
                     analysis_id: str = "analysis",
                     ext: Optional[str] = None,
                     subdir: Optional[str] = None
                     ) -> Dict[str, Path]:
        """Save every figure in ``figures`` (key → Figure). Returns key→Path."""
        if subdir is None:
            subdir = f"{_slug(analysis_id)}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        paths: Dict[str, Path] = {}
        for key, fig in figures.items():
            if fig is None:
                continue
            try:
                paths[key] = self.export_figure(
                    fig, plot_key=key, analysis_id=analysis_id,
                    ext=ext, subdir=subdir)
            except Exception as exc:
                # Skip one bad figure — keep going
                paths[key] = Path(f"ERROR: {exc}")
        return paths

    # ------------------------------------------------------------------
    def write_html_index(self, paths: Mapping[str, Path],
                         title: str = "GeneVariate exports",
                         subdir: Optional[str] = None) -> Path:
        """Write an HTML gallery that references images in ``paths``."""
        out_dir = (self.base_dir / _slug(subdir)) if subdir else self.base_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = out_dir / "index.html"
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        body = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>{_html.escape(title)}</title>",
            "<style>",
            "body{font-family:Segoe UI,sans-serif;background:#EAF6FE;color:#0E2A45;margin:32px;}",
            "h1{color:#0A5B9A;font-weight:600;}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px;}",
            ".card{background:#fff;border:1px solid #C5DAEA;border-radius:8px;padding:10px;box-shadow:0 1px 3px rgba(30,144,224,.1);} ",
            ".card img{max-width:100%;border-radius:4px;}",
            ".card .k{font-weight:600;color:#0A5B9A;margin-bottom:4px;}",
            ".ts{color:#5F7D95;font-size:12px;}",
            "</style></head><body>",
            f"<h1>{_html.escape(title)}</h1>",
            f"<div class='ts'>Generated {ts}</div>",
            "<div class='grid'>",
        ]
        for key, path in paths.items():
            rel = Path(path).name if Path(path).parent == out_dir else str(path)
            is_img = Path(path).suffix.lower() in (".png", ".jpg", ".jpeg", ".svg")
            body.append("<div class='card'>")
            body.append(f"<div class='k'>{_html.escape(str(key))}</div>")
            if is_img:
                body.append(f"<a href='{_html.escape(str(rel))}'>"
                            f"<img src='{_html.escape(str(rel))}'></a>")
            else:
                body.append(f"<a href='{_html.escape(str(rel))}'>"
                            f"{_html.escape(str(rel))}</a>")
            body.append("</div>")
        body.append("</div></body></html>")
        idx.write_text("\n".join(body), encoding="utf-8")
        return idx
