"""
Interactive Subset Analyzer Window.
Provides comprehensive analysis of classified sample subsets with PCA and DPC.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from datetime import datetime
from pathlib import Path
from genevariate.gui.theme import AERO, ensure_theme, style_toolbar, style_window

# Unified GeneVariate plot stylesheet
try:
    from genevariate.utils.viz_style import (
        apply_genevariate_style as _apply_gv_style,
        palette_for as _palette_for,
        cmap_for as _cmap_for,
        smart_figsize as _smart_figsize,
        cap_figsize as _cap_figsize,
        make_interactive as _make_interactive,
        attach_point_labels as _point_labels,
        EXPORT_DPI as _EXPORT_DPI,
    )
    _apply_gv_style()
except Exception:
    _EXPORT_DPI = 300
    def _palette_for(n, use_case="discrete"):
        import matplotlib.colors as _mc
        return [_mc.to_hex(c) for c in sns.color_palette("tab10", n)]
    def _cmap_for(kind="sequential"):
        return "viridis" if kind != "diverging" else "RdBu_r"
    def _smart_figsize(kind="default"): return (10, 6)
    def _cap_figsize(w, h, max_w=16.0, max_h=10.0):
        return (min(w, max_w), min(h, max_h))
    def _make_interactive(fig, **kw): return None
    def _point_labels(artist, labels): return None



def _lim(key):
    """How much this window shows, as the user set it."""
    from genevariate.gui import display_limits
    n = display_limits.get(key)
    return 10 ** 9 if n is None else n


class ScrollableCanvasFrame(ttk.Frame):
    """A scrollable frame for displaying large content."""
    
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # Create canvas and scrollbars
        self.canvas = tk.Canvas(self, bg=AERO["bg_top"], highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        # Pack scrollbars and canvas
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create inner frame
        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor=tk.NW)
        
        # Bind events
        self.inner_frame.bind('<Configure>', self.on_frame_configure)
        self.canvas.bind('<Configure>', self.on_canvas_configure)
    
    def on_frame_configure(self, event=None):
        """Update scrollregion when inner frame changes size."""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
    
    def on_canvas_configure(self, event):
        """Update inner frame width when canvas changes size."""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)


#: Separator between the sample id and its group in a point's hover label.
_LABEL_SEP = " · "


class InteractiveSubsetAnalyzerWindow(tk.Toplevel):
    """
    Interactive window for analyzing sample subsets.
    Provides PCA, DPC, and distribution analysis.
    COMPLETE VERSION - NO SIMPLIFICATIONS.
    """
    
    def __init__(self, parent, app_ref, step2_dataframe, source_description=""):
        super().__init__(parent)
        ensure_theme(self)
        style_window(self)
        # Stay hidden until there is something to show. Mapping first meant
        # that unclassified data flashed a full-screen empty window behind the
        # warning, which reads as a crash rather than as a precondition.
        self.withdraw()

        self.app_ref = app_ref
        self.step2_df = step2_dataframe.copy()
        self.source_desc = source_description
        
        self.title(f"Interactive Subset Analyzer - {source_description}")
        self.geometry("1400x900")
        
        # State
        self.current_grouping_col = None
        self.current_plot_type = 'bar'
        self.filtered_df = self.step2_df.copy()
        self.figures = {}
        
        # Find classification columns
        self.classification_cols = [c for c in self.step2_df.columns 
                                    if c.startswith('Classified_')]
        
        if not self.classification_cols:
            messagebox.showwarning(
                "No Classification",
                "No classification columns found.\nMake sure data has been classified first.",
                parent=parent
            )
            self.destroy()
            return

        # Set default grouping
        self.current_grouping_col = self.classification_cols[0]

        self._setup_ui()
        self._update_display()
        self.deiconify()
    
    def _setup_ui(self):
        """Setup the user interface - COMPLETE."""
        # Top control panel
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Grouping selection
        ttk.Label(control_frame, text="Group by:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.grouping_var = tk.StringVar(value=self.current_grouping_col)
        grouping_menu = ttk.Combobox(control_frame, textvariable=self.grouping_var, 
                                     values=self.classification_cols, state='readonly', width=25)
        grouping_menu.pack(side=tk.LEFT, padx=5)
        grouping_menu.bind('<<ComboboxSelected>>', self._on_grouping_changed)
        
        # Filter entry
        ttk.Label(control_frame, text="Filter:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(20, 5))
        
        self.filter_var = tk.StringVar()
        self.filter_entry = ttk.Entry(control_frame, textvariable=self.filter_var, width=30)
        self.filter_entry.pack(side=tk.LEFT, padx=5)
        self.filter_entry.bind('<Return>', lambda e: self._apply_filter())
        
        ttk.Button(control_frame, text="Apply Filter", command=self._apply_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear Filter", command=self._clear_filter).pack(side=tk.LEFT, padx=5)
        
        # Export button
        ttk.Button(control_frame, text="Export All", command=self._export_data).pack(side=tk.RIGHT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text=f"Total: {len(self.step2_df)} samples", 
                                      foreground='blue', font=('Segoe UI', 9, 'italic'))
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Distribution Overview
        self.overview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_tab, text=" Distribution Overview ")
        
        # Tab 2: PCA Analysis
        self.pca_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pca_tab, text=" PCA Analysis ")
        
        # Tab 3: DPC Analysis
        self.dpc_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dpc_tab, text=" Density Peak Clustering ")
        
        # Tab 4: Data Table
        self.table_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.table_tab, text=" Data Table ")
        
        self._setup_overview_tab()
        self._setup_pca_tab()
        self._setup_dpc_tab()
        self._setup_table_tab()
    
    def _setup_overview_tab(self):
        """Setup distribution overview tab."""
        # Controls
        ctrl_frame = ttk.Frame(self.overview_tab)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ctrl_frame, text="Plot type:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.plot_type_var = tk.StringVar(value='bar')
        for ptype in [('Bar Chart', 'bar'), ('Pie Chart', 'pie')]:
            ttk.Radiobutton(ctrl_frame, text=ptype[0], variable=self.plot_type_var, 
                          value=ptype[1], command=self._update_overview).pack(side=tk.LEFT, padx=5)
        
        # Canvas area
        self.overview_canvas_frame = ttk.Frame(self.overview_tab)
        self.overview_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _setup_pca_tab(self):
        """Setup PCA analysis tab."""
        # Info label
        info = ttk.Label(self.pca_tab, 
                        text="PCA (Principal Component Analysis) - Dimensionality reduction visualization",
                        font=('Segoe UI', 9, 'italic'), foreground='gray')
        info.pack(pady=5)
        
        # Controls
        ctrl_frame = ttk.Frame(self.pca_tab)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(ctrl_frame, text="Compute PCA", command=self._compute_pca).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Save Plot", command=lambda: self._save_current_plot('pca')).pack(side=tk.LEFT, padx=5)
        
        # Canvas area
        self.pca_canvas_frame = ttk.Frame(self.pca_tab)
        self.pca_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Placeholder
        ttk.Label(self.pca_canvas_frame, 
                 text="Click 'Compute PCA' to generate analysis",
                 font=('Segoe UI', 11), foreground='gray').pack(expand=True)
    
    def _setup_dpc_tab(self):
        """Setup DPC analysis tab."""
        # Info label
        info = ttk.Label(self.dpc_tab,
                        text="DPC (Density Peak Clustering) - Automatic cluster detection",
                        font=('Segoe UI', 9, 'italic'), foreground='gray')
        info.pack(pady=5)
        
        # Controls
        ctrl_frame = ttk.Frame(self.dpc_tab)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(ctrl_frame, text="Compute DPC", command=self._compute_dpc).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Save Plot", command=lambda: self._save_current_plot('dpc')).pack(side=tk.LEFT, padx=5)
        
        # Canvas area
        self.dpc_canvas_frame = ttk.Frame(self.dpc_tab)
        self.dpc_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Placeholder
        ttk.Label(self.dpc_canvas_frame,
                 text="Click 'Compute DPC' to generate analysis",
                 font=('Segoe UI', 11), foreground='gray').pack(expand=True)
    
    def _setup_table_tab(self):
        """Setup data table tab."""
        # Controls
        ctrl_frame = ttk.Frame(self.table_tab)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(ctrl_frame, text="Copy to Clipboard", command=self._copy_table).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Export CSV", command=self._export_table).pack(side=tk.LEFT, padx=5)
        
        # Table
        table_container = ttk.Frame(self.table_tab)
        table_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview with scrollbars
        tree_scroll_y = ttk.Scrollbar(table_container)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.data_tree = ttk.Treeview(table_container, 
                                      yscrollcommand=tree_scroll_y.set,
                                      xscrollcommand=tree_scroll_x.set)
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tree_scroll_y.config(command=self.data_tree.yview)
        tree_scroll_x.config(command=self.data_tree.xview)

    def _update_display(self):
            """Update all displays with current data."""
            self._update_overview()
            self._update_table()
            self._update_status()
    
    def _update_status(self):
        """Update status label."""
        total = len(self.step2_df)
        filtered = len(self.filtered_df)
        
        if filtered < total:
            self.status_label.config(
                text=f"Showing: {filtered} / {total} samples (filtered)",
                foreground='orange'
            )
        else:
            self.status_label.config(
                text=f"Total: {total} samples",
                foreground='blue'
            )
    
    def _open_sample_card(self, pick):
        """Show the full record for the PCA point the user clicked.

        A PCA plot can only show two coordinates, so an outlier raises a
        question it cannot answer. Clicking it answers it.
        """
        from genevariate.gui.windows.sample_card import show_sample_card

        gsm = str(pick.label).split(_LABEL_SEP)[0].strip()
        frame = self.filtered_df
        # The sample id lives in the index here, but the card looks it up by
        # column, so surface it as one.
        if "GSM" not in frame.columns:
            frame = frame.reset_index().rename(columns={"index": "GSM"})
        show_sample_card(self, gsm, frame)

    def _on_grouping_changed(self, event=None):
        """Handle grouping column change."""
        self.current_grouping_col = self.grouping_var.get()
        self._update_display()
    
    def _apply_filter(self):
        """Apply filter to data."""
        filter_text = self.filter_var.get().strip().lower()
        
        if not filter_text:
            self.filtered_df = self.step2_df.copy()
            self._update_display()
            return
        
        # Filter across all text columns
        mask = pd.Series([False] * len(self.step2_df))
        
        for col in self.step2_df.columns:
            if self.step2_df[col].dtype == 'object':
                mask |= self.step2_df[col].astype(str).str.lower().str.contains(filter_text, na=False)
        
        self.filtered_df = self.step2_df[mask].copy()
        
        if self.filtered_df.empty:
            messagebox.showinfo("No Results", f"No samples match filter: '{filter_text}'", parent=self)
            self.filtered_df = self.step2_df.copy()
        
        self._update_display()
    
    def _clear_filter(self):
        """Clear filter."""
        self.filter_var.set('')
        self.filtered_df = self.step2_df.copy()
        self._update_display()
    
    def _update_overview(self):
        """Update overview tab with current plot type."""
        # Clear previous
        for widget in self.overview_canvas_frame.winfo_children():
            widget.destroy()
        
        if self.current_grouping_col not in self.filtered_df.columns:
            ttk.Label(self.overview_canvas_frame, 
                     text="No grouping column available",
                     font=('Segoe UI', 11), foreground='red').pack(expand=True)
            return
        
        # Count samples per group
        full_counts = self.filtered_df[self.current_grouping_col].value_counts()

        if full_counts.empty:
            ttk.Label(self.overview_canvas_frame,
                     text="No data to display",
                     font=('Segoe UI', 11), foreground='gray').pack(expand=True)
            return

        # Scalability: a whole-platform label column can carry thousands of
        # distinct values. Drawing a bar/wedge per value is unreadable and
        # slow, so keep the top categories by count and fold the long tail
        # into a single "Other" entry. How many is the user's setting.
        n_total = len(full_counts)
        truncated = n_total > _lim('categories')
        if truncated:
            counts = full_counts.iloc[:_lim('categories')].copy()
            counts[f"Other ({n_total - _lim('categories')} more)"] = int(
                full_counts.iloc[_lim('categories'):].sum())
        else:
            counts = full_counts
        title = f'Sample Distribution by {self.current_grouping_col}'
        if truncated:
            from genevariate.gui import display_limits
            title += "  (%s)" % display_limits.cap_note(len(counts) - 1, n_total,
                                                        noun="categories")

        # Create figure
        fig = Figure(figsize=(10, 6))
        ax = fig.subplots()

        plot_type = self.plot_type_var.get()

        if plot_type == 'bar':
            counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel(self.current_grouping_col, fontsize=12, fontweight='bold')
            ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            # This figure is a bare Figure(), never registered with pyplot, so
            # plt.xticks() rotated the labels of some unrelated current figure
            # - and created an empty one when there was none, leaking a canvas
            # on every redraw. Rotate this axis's own labels.
            for _lbl in ax.get_xticklabels():
                _lbl.set_rotation(45)
                _lbl.set_horizontalalignment('right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # Add count labels on bars
            for i, (label, value) in enumerate(counts.items()):
                ax.text(i, value + max(counts) * 0.02, str(value),
                       ha='center', va='bottom', fontweight='bold')

        elif plot_type == 'pie':
            n_cats = len(counts)
            colors = _palette_for(n_cats)
            if n_cats > 10:
                # Too many categories for inline labels - use legend instead
                wedges, texts, autotexts = ax.pie(
                    counts, labels=None, autopct='%1.1f%%',
                    colors=colors, startangle=90, pctdistance=0.80)
                ax.legend(wedges, [str(l)[:25] for l in counts.index],
                          loc='center left', bbox_to_anchor=(1.0, 0.5),
                          fontsize=max(6, 9 - n_cats // 10))
                fig.subplots_adjust(right=0.65)
                for autotext in autotexts:
                    autotext.set_fontsize(max(5, 9 - n_cats // 8))
                    autotext.set_color('black')
            else:
                wedges, texts, autotexts = ax.pie(
                    counts, labels=counts.index, autopct='%1.1f%%',
                    colors=colors, startangle=90)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.overview_canvas_frame)
        canvas.draw()
        _make_interactive(fig)
        canvas_widget = canvas.get_tk_widget()
        
        toolbar = NavigationToolbar2Tk(canvas, self.overview_canvas_frame)
        toolbar.update()
        style_toolbar(toolbar)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.figures['overview'] = fig
    
    def _update_table(self):
        """Update data table."""
        # Clear existing
        self.data_tree.delete(*self.data_tree.get_children())
        
        # Get columns to display (limit for performance)
        display_cols = list(self.filtered_df.columns)[:20]
        
        # Configure columns
        self.data_tree['columns'] = display_cols
        self.data_tree['show'] = 'headings'
        
        for col in display_cols:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=120, anchor=tk.W)
        
        # Add rows (limit to first 1000 for performance)
        for idx, row in self.filtered_df.head(1000).iterrows():
            values = [str(row[col])[:50] for col in display_cols]
            self.data_tree.insert('', tk.END, values=values)
        
        if len(self.filtered_df) > 1000:
            self.data_tree.insert('', tk.END, values=['...'] * len(display_cols))
    
    def _compute_pca(self):
        """Compute and display PCA analysis."""
        # Clear previous
        for widget in self.pca_canvas_frame.winfo_children():
            widget.destroy()
        
        # Show progress
        progress_label = ttk.Label(self.pca_canvas_frame, 
                                  text="Computing PCA...", 
                                  font=('Segoe UI', 11, 'bold'))
        progress_label.pack(expand=True)
        self.update_idletasks()
        
        try:
            # Get numeric columns only
            numeric_cols = self.filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove metadata columns
            from genevariate.config import METADATA_EXCLUSIONS
            # The exclusion list is mixed case ('Title', 'series_id', ...), so
            # comparing an upper-cased column against it only ever matched the
            # three entries that are already upper ('GSM', 'ID', 'GPL') and let
            # every other metadata column through as if it were a gene.
            exclude_upper = {s.upper() for s in METADATA_EXCLUSIONS}
            gene_cols = [c for c in numeric_cols if c.upper() not in exclude_upper]

            if len(gene_cols) < 2:
                progress_label.destroy()
                ttk.Label(self.pca_canvas_frame,
                         text="Not enough numeric columns for PCA\n(Need at least 2 gene expression columns)",
                         font=('Segoe UI', 11), foreground='red').pack(expand=True)
                return
            
            # Complete case: a sample that did not measure one of these genes
            # is left out, not entered at zero. Zero is a real expression
            # level, so filling with it puts the unmeasured samples together
            # in one corner of the projection and draws that as structure.
            _sub = self.filtered_df[gene_cols]
            _ok = _sub.notna().all(axis=1).to_numpy()
            _n_drop = int((~_ok).sum())
            X = _sub[_ok].values
            
            if len(X) < 3:
                progress_label.destroy()
                ttk.Label(self.pca_canvas_frame,
                         text="Not enough samples for PCA\n(Need at least 3 samples)",
                         font=('Segoe UI', 11), foreground='red').pack(expand=True)
                return
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            progress_label.destroy()
            
            # Create plot
            fig = Figure(figsize=(10, 8))
            ax = fig.subplots()

            # Name each point so hovering it reports the sample it came from
            sample_ids = list(self.filtered_df.index.astype(str))

            # Color by grouping column
            if self.current_grouping_col in self.filtered_df.columns:
                groups = pd.Series(self.filtered_df[self.current_grouping_col].values)
                # Scalability: at whole-platform scale a label column can hold
                # thousands of distinct values. One scatter + legend entry each
                # would be unreadable and slow, and tab10 only has 10 colours.
                # Colour the top groups by count, fold the rest into one grey
                # "Other" series, and legend only the named groups. How many
                # are named is the user's setting.
                order = groups.value_counts()
                named = list(order.index[:_lim('groups')])
                palette = _palette_for(len(named))
                gvals = groups.to_numpy()
                for i, group in enumerate(named):
                    mask = gvals == group
                    sc = ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                             c=[palette[i]], label=str(group), s=60, alpha=0.6,
                             edgecolors='black', linewidth=0.5)
                    _point_labels(sc, [f"{sample_ids[j]}{_LABEL_SEP}{group}"
                                       for j in np.where(mask)[0]])
                if len(order) > _lim('groups'):
                    rest = ~np.isin(gvals, named)
                    n_rest = len(order) - _lim('groups')
                    sc = ax.scatter(X_pca[rest, 0], X_pca[rest, 1],
                             c='#B0B0B0', label=f'Other ({n_rest} more)', s=40,
                             alpha=0.4, edgecolors='none')
                    _point_labels(sc, [f"{sample_ids[j]}{_LABEL_SEP}{gvals[j]}"
                                       for j in np.where(rest)[0]])

                has_legend = True
            else:
                sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c='steelblue', s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
                _point_labels(sc, sample_ids)
                has_legend = False

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                         fontsize=12, fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
                         fontsize=12, fontweight='bold')
            ax.set_title('PCA: Principal Component Analysis', fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')

            fig.tight_layout()
            if has_legend:
                fig.subplots_adjust(right=0.72)
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
                          frameon=True, shadow=True)
            
            # Embed
            canvas = FigureCanvasTkAgg(fig, master=self.pca_canvas_frame)
            canvas.draw()
            _make_interactive(fig, on_pick=self._open_sample_card)
            canvas_widget = canvas.get_tk_widget()
            
            toolbar = NavigationToolbar2Tk(canvas, self.pca_canvas_frame)
            toolbar.update()
            style_toolbar(toolbar)
            toolbar.pack(side=tk.TOP, fill=tk.X)
            
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            self.figures['pca'] = fig
            
        except Exception as e:
            progress_label.destroy()
            import traceback
            error_msg = traceback.format_exc()
            print(error_msg)
            ttk.Label(self.pca_canvas_frame,
                     text=f"Error computing PCA:\n{str(e)}",
                     font=('Segoe UI', 10), foreground='red').pack(expand=True)

    def _compute_dpc(self):
            """Compute and display Density Peak Clustering."""
            # Clear previous
            for widget in self.dpc_canvas_frame.winfo_children():
                widget.destroy()
            
            # Show progress
            progress_label = ttk.Label(self.dpc_canvas_frame,
                                      text="Computing DPC...",
                                      font=('Segoe UI', 11, 'bold'))
            progress_label.pack(expand=True)
            self.update_idletasks()
            
            try:
                # Get numeric columns
                numeric_cols = self.filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                
                from genevariate.config import METADATA_EXCLUSIONS
                # Same case mismatch as in the PCA path above.
                exclude_upper = {s.upper() for s in METADATA_EXCLUSIONS}
                gene_cols = [c for c in numeric_cols if c.upper() not in exclude_upper]
                
                if len(gene_cols) < 2:
                    progress_label.destroy()
                    ttk.Label(self.dpc_canvas_frame,
                             text="Not enough numeric columns for DPC",
                             font=('Segoe UI', 11), foreground='red').pack(expand=True)
                    return
                
                # Complete case here too - see the PCA tab: an unmeasured gene
                # is not an observation of zero.
                _sub = self.filtered_df[gene_cols]
                _ok = _sub.notna().all(axis=1).to_numpy()
                _n_drop = int((~_ok).sum())
                X = _sub[_ok].values
                
                if len(X) < 10:
                    progress_label.destroy()
                    ttk.Label(self.dpc_canvas_frame,
                             text="Not enough samples for DPC\n(Need at least 10 samples)",
                             font=('Segoe UI', 11), foreground='red').pack(expand=True)
                    return
                
                # Standardize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Compute pairwise distances
                from scipy.spatial.distance import pdist, squareform
                distances = squareform(pdist(X_scaled, metric='euclidean'))
                
                # Compute local density
                dc = np.percentile(distances[distances > 0], 2)  # 2% percentile as cutoff
                rho = np.sum(distances < dc, axis=1) - 1  # -1 to exclude self
                
                # Compute minimum distance to higher density point
                delta = np.zeros(len(X))
                for i in range(len(X)):
                    higher_density_points = np.where(rho > rho[i])[0]
                    if len(higher_density_points) > 0:
                        delta[i] = np.min(distances[i, higher_density_points])
                    else:
                        delta[i] = np.max(distances[i, :])
                
                progress_label.destroy()
                
                # Create decision graph
                fig = Figure(figsize=(14, 6))
                ax1, ax2 = fig.subplots(1, 2)
                
                # Decision graph
                ax1.scatter(rho, delta, c='steelblue', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
                ax1.set_xlabel('Density (ρ)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Distance to higher density (δ)', fontsize=12, fontweight='bold')
                ax1.set_title('DPC Decision Graph', fontsize=13, fontweight='bold')
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # Highlight potential cluster centers (high rho and high delta)
                gamma = rho * delta
                threshold = np.percentile(gamma, 95)
                centers = gamma > threshold
                ax1.scatter(rho[centers], delta[centers], c='red', s=150, marker='*', 
                           edgecolors='black', linewidth=1, label='Potential Centers', zorder=10)
                ax1.legend()
                
                # 2D projection (PCA)
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # Assign clusters based on density and distance
                n_clusters = np.sum(centers)
                if n_clusters > 0:
                    cluster_centers_idx = np.where(centers)[0]
                    labels = np.zeros(len(X), dtype=int)
                    
                    for i in range(len(X)):
                        if i in cluster_centers_idx:
                            labels[i] = list(cluster_centers_idx).index(i)
                        else:
                            # Assign to nearest higher-density cluster center
                            higher = np.where(rho > rho[i])[0]
                            if len(higher) > 0:
                                nearest = higher[np.argmin(distances[i, higher])]
                                if nearest in cluster_centers_idx:
                                    labels[i] = list(cluster_centers_idx).index(nearest)
                    
                    # Plot clusters
                    n_clust = int(max(labels)) + 1
                    colors = _palette_for(n_clust)
                    for cluster_id in range(n_clust):
                        mask = labels == cluster_id
                        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                                  c=[colors[cluster_id]], label=f'Cluster {cluster_id + 1}',
                                  s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
                    
                    # Mark centers
                    ax2.scatter(X_pca[centers, 0], X_pca[centers, 1],
                               c='red', s=200, marker='*', edgecolors='black', linewidth=1.5,
                               label='Centers', zorder=10)
                    
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', s=60, alpha=0.6)
                    ax2.text(0.5, 0.5, 'No clear clusters detected', 
                            transform=ax2.transAxes, ha='center', fontsize=12, color='red')
                
                ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                              fontsize=12, fontweight='bold')
                ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                              fontsize=12, fontweight='bold')
                ax2.set_title('DPC Clustering Result', fontsize=13, fontweight='bold')
                ax2.grid(True, alpha=0.3, linestyle='--')
                
                fig.tight_layout()

                # Embed
                canvas = FigureCanvasTkAgg(fig, master=self.dpc_canvas_frame)
                canvas.draw()
                _make_interactive(fig)
                canvas_widget = canvas.get_tk_widget()
                
                toolbar = NavigationToolbar2Tk(canvas, self.dpc_canvas_frame)
                toolbar.update()
                style_toolbar(toolbar)
                toolbar.pack(side=tk.TOP, fill=tk.X)
                
                canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                
                self.figures['dpc'] = fig
                
            except Exception as e:
                progress_label.destroy()
                import traceback
                error_msg = traceback.format_exc()
                print(error_msg)
                ttk.Label(self.dpc_canvas_frame,
                         text=f"Error computing DPC:\n{str(e)}",
                         font=('Segoe UI', 10), foreground='red').pack(expand=True)
        
    def _save_current_plot(self, plot_type):
        """Save current plot to file."""
        if plot_type not in self.figures:
            messagebox.showwarning("No Plot", "No plot to save. Generate the plot first.", parent=self)
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ],
            initialfile=f"{plot_type}_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if filepath:
            try:
                self.figures[plot_type].savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to:\n{filepath}", parent=self)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot:\n{e}", parent=self)
    
    def _copy_table(self):
        """Copy table data to clipboard."""
        try:
            self.filtered_df.to_clipboard(index=False)
            messagebox.showinfo("Success", "Data copied to clipboard!", parent=self)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy:\n{e}", parent=self)
    
    def _export_table(self):
        """Export table data to CSV."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"subset_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if filepath:
            try:
                self.filtered_df.to_csv(filepath, index=False)
                messagebox.showinfo("Success", f"Data exported to:\n{filepath}", parent=self)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export:\n{e}", parent=self)
    
    def _export_data(self):
        """Write the subset, every table and every plot this window has drawn."""
        from pathlib import Path
        from genevariate.gui.exporting import export_window
        d = filedialog.askdirectory(title="Export folder", parent=self)
        if not d:
            return
        out = Path(d)
        out.mkdir(parents=True, exist_ok=True)
        for key, compute in (('pca', self._compute_pca), ('dpc', self._compute_dpc)):
            if key not in self.figures:
                try:
                    compute()
                except Exception:
                    pass
        written = export_window(self, out, figures=self.figures, prefix="subset_")
        try:
            self.filtered_df.to_csv(out / "subset_data.csv", index=False)
            written.append(out / "subset_data.csv")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export:\n{e}", parent=self)
            return
        messagebox.showinfo("Exported",
                            f"Saved {len(written)} file(s) to:\n{out}",
                            parent=self)

    def destroy(self):
            """Clean up before closing."""
            # Close all figures
            for fig in self.figures.values():
                try:
                    plt.close(fig)
                except:
                    pass
            
            super().destroy()
