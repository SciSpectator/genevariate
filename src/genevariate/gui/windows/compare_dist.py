"""
Distribution Comparison Windows.
Provides tools for comparing gene expression distributions across groups.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.stats import ranksums, wasserstein_distance
from datetime import datetime
from pathlib import Path


class CompareDistributionsWindow(tk.Toplevel):
    """
    Window for loading and comparing distributions from CSV files.
    Simple version for loading external data.
    """
    
    def __init__(self, parent, app_ref):
        super().__init__(parent)
        
        self.app_ref = app_ref
        self.title("Compare Distributions")
        self.geometry("900x700")
        self.transient(parent)
        
        self.loaded_files = {}
        self.current_figure = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup user interface."""
        # Instructions
        inst_frame = ttk.Frame(self)
        inst_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            inst_frame,
            text="Load CSV files with classified samples to compare distributions",
            font=('Segoe UI', 10, 'italic'),
            foreground='gray'
        ).pack()
        
        # File loading
        load_frame = ttk.LabelFrame(self, text="Load Files", padding=10)
        load_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(load_frame, text="📁 Load CSV File", command=self._load_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="🗑️ Clear All", command=self._clear_files).pack(side=tk.LEFT, padx=5)
        
        # Loaded files list
        list_frame = ttk.LabelFrame(self, text="Loaded Files", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=8)
        scrollbar = ttk.Scrollbar(list_frame, command=self.file_listbox.yview)
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Comparison controls
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(ctrl_frame, text="Gene:").pack(side=tk.LEFT, padx=5)
        self.gene_entry = ttk.Entry(ctrl_frame, width=20)
        self.gene_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(ctrl_frame, text="📊 Compare", command=self._compare).pack(side=tk.LEFT, padx=5)
        
        # Canvas for plots
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(
            self.canvas_frame,
            text="Load files and click 'Compare' to begin",
            font=('Segoe UI', 11),
            foreground='gray'
        ).pack(expand=True)
    
    def _load_file(self):
        """Load a CSV file."""
        filepath = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("Compressed CSV", "*.csv.gz"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            compression = 'gzip' if filepath.endswith('.gz') else None
            df = pd.read_csv(filepath, compression=compression, low_memory=False)
            
            filename = Path(filepath).name
            self.loaded_files[filename] = df
            
            self.file_listbox.insert(tk.END, f"{filename} ({len(df)} samples)")
            
            messagebox.showinfo("Success", f"Loaded {len(df)} samples from {filename}", parent=self)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}", parent=self)
    
    def _clear_files(self):
        """Clear all loaded files."""
        if not self.loaded_files:
            return
        
        if messagebox.askyesno("Confirm", "Clear all loaded files?", parent=self):
            self.loaded_files.clear()
            self.file_listbox.delete(0, tk.END)
    
    def _compare(self):
        """Compare distributions."""
        if not self.loaded_files:
            messagebox.showwarning("No Files", "Please load at least one CSV file first.", parent=self)
            return
        
        gene = self.gene_entry.get().strip().upper()
        
        if not gene:
            messagebox.showwarning("No Gene", "Please enter a gene symbol.", parent=self)
            return
        
        # Clear previous plot
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # Find gene column in each file
        data_dict = {}
        
        for filename, df in self.loaded_files.items():
            # Look for gene column (case-insensitive)
            gene_col = None
            for col in df.columns:
                if col.upper() == gene:
                    gene_col = col
                    break
            
            if gene_col:
                data_dict[filename] = df[gene_col].dropna().astype(float)
        
        if not data_dict:
            messagebox.showinfo("Not Found", f"Gene '{gene}' not found in any loaded files.", parent=self)
            return
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (name, data) in enumerate(data_dict.items()):
            ax.hist(data, bins=30, alpha=0.5, label=name, edgecolor='black')
        
        ax.set_xlabel('Expression', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Distribution Comparison: {gene}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        
        toolbar = NavigationToolbar2Tk(canvas, self.canvas_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.current_figure = fig


class CustomCompareWindow(tk.Toplevel):
    """
    Advanced comparison window with statistical tests.
    Used for comparing regions selected from gene distribution explorer.
    COMPLETE VERSION - NO SIMPLIFICATIONS.
    """
    
    def __init__(self, parent, app_ref, df_full, data_map, bg_map, grp_gsm_map, 
                 title_suffix="", grouping_col="Classified_Condition"):
        super().__init__(parent)
        
        self.app_ref = app_ref
        self.df_full = df_full.copy()
        self.data_map = data_map
        self.bg_map = bg_map
        self.grp_gsm_map = grp_gsm_map
        self.grouping_col = grouping_col
        
        self.title(f"Advanced Distribution Comparison - {title_suffix}")
        self.geometry("1200x800")
        self.transient(parent)
        
        self.current_figures = {}
        self.show_rugs = True
        
        self._setup_ui()
        self._plot_distributions()
    
    def _setup_ui(self):
        """Setup user interface - COMPLETE."""
        # Top controls
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Grouping selection
        ttk.Label(ctrl_frame, text="Group by:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Find classification columns
        class_cols = [c for c in self.df_full.columns if c.startswith('Classified_')]
        
        if class_cols:
            self.grouping_var = tk.StringVar(value=self.grouping_col if self.grouping_col in class_cols else class_cols[0])
            
            grouping_menu = ttk.Combobox(ctrl_frame, textvariable=self.grouping_var,
                                        values=class_cols, state='readonly', width=25)
            grouping_menu.pack(side=tk.LEFT, padx=5)
            grouping_menu.bind('<<ComboboxSelected>>', lambda e: self._update_grouping())
        else:
            ttk.Label(ctrl_frame, text="No classification columns", foreground='red').pack(side=tk.LEFT, padx=5)
        
        # Display options
        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.rug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Show Rug Plot", variable=self.rug_var,
                       command=self._toggle_rugs).pack(side=tk.LEFT, padx=5)
        
        self.kde_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Show KDE", variable=self.kde_var,
                       command=self._toggle_kde).pack(side=tk.LEFT, padx=5)
        
        # Export
        ttk.Button(ctrl_frame, text="💾 Save Plot", command=self._save_plot).pack(side=tk.RIGHT, padx=5)
        ttk.Button(ctrl_frame, text="📊 Export Stats", command=self._export_stats).pack(side=tk.RIGHT, padx=5)
        
        # Notebook for different views
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Overlaid distributions
        self.overlay_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overlay_tab, text="📊 Overlaid Distributions")
        
        # Tab 2: Side-by-side
        self.sidebyside_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sidebyside_tab, text="📈 Side-by-Side")
        
        # Tab 3: Statistics
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="📋 Statistical Tests")
        
        self._setup_stats_tab()

    def _setup_stats_tab(self):
        """Setup statistics tab."""
        # Info label
        ttk.Label(
            self.stats_tab,
            text="Statistical comparisons between groups",
            font=('Segoe UI', 10, 'italic'),
            foreground='gray'
        ).pack(pady=10)
        
        # Treeview for stats
        tree_frame = ttk.Frame(self.stats_tab)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ('Group 1', 'Group 2', 'Test', 'Statistic', 'P-value', 'Effect Size')
        self.stats_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, width=150, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(tree_frame, command=self.stats_tree.yview)
        self.stats_tree.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def _plot_distributions(self):
        """Plot distribution comparisons - COMPLETE."""
        # Clear previous
        for widget in self.overlay_tab.winfo_children():
            widget.destroy()
        for widget in self.sidebyside_tab.winfo_children():
            widget.destroy()
        
        if not self.data_map:
            ttk.Label(self.overlay_tab, text="No data to plot", font=('Segoe UI', 11)).pack(expand=True)
            return
        
        # Create overlay plot
        fig_overlay, ax_overlay = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.data_map)))
        
        for i, (group_name, values) in enumerate(self.data_map.items()):
            values_clean = values.dropna()
            
            if len(values_clean) < 2:
                continue
            
            # Histogram
            ax_overlay.hist(values_clean, bins=30, alpha=0.4, label=group_name,
                          color=colors[i], edgecolor='black', linewidth=0.5)
            
            # KDE if enabled
            if self.kde_var.get() and len(values_clean) > 5:
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(values_clean)
                    x_range = np.linspace(values_clean.min(), values_clean.max(), 200)
                    kde_vals = kde(x_range)

                    # Scale KDE to match histogram counts:
                    # histogram bin_width = (max - min) / n_bins
                    # scaled_kde = kde_density * n_samples * bin_width
                    n_bins = 30
                    data_range = values_clean.max() - values_clean.min()
                    if data_range < 1e-12:
                        # Constant data — skip KDE (would produce invisible line)
                        raise ValueError("constant data")
                    bin_width = data_range / n_bins
                    kde_scaled = kde_vals * len(values_clean) * bin_width

                    ax_overlay.plot(x_range, kde_scaled,
                                  color=colors[i], linewidth=2, linestyle='--', alpha=0.8)
                except Exception:
                    pass
            
            # Rug plot if enabled
            if self.rug_var.get():
                rug_height = ax_overlay.get_ylim()[1] * 0.02
                for val in values_clean.sample(min(100, len(values_clean))):
                    ax_overlay.plot([val, val], [0, rug_height], color=colors[i], 
                                  alpha=0.3, linewidth=1)
        
        ax_overlay.set_xlabel('Expression Value', fontsize=12, fontweight='bold')
        ax_overlay.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax_overlay.set_title('Overlaid Distribution Comparison', fontsize=14, fontweight='bold', pad=15)
        ax_overlay.grid(True, alpha=0.3, linestyle='--')

        fig_overlay.tight_layout()
        fig_overlay.subplots_adjust(right=0.75)
        ax_overlay.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        
        # Embed overlay
        canvas_overlay = FigureCanvasTkAgg(fig_overlay, master=self.overlay_tab)
        canvas_overlay.draw()
        
        toolbar_overlay = NavigationToolbar2Tk(canvas_overlay, self.overlay_tab)
        toolbar_overlay.update()
        toolbar_overlay.pack(side=tk.TOP, fill=tk.X)
        
        canvas_overlay.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.current_figures['overlay'] = fig_overlay
        
        # Create side-by-side plot
        n_groups = len(self.data_map)
        # Cap width so figures don't become unreadably wide
        fig_width = min(5 * n_groups, 30)  # max 30 inches
        fig_side, axes = plt.subplots(1, n_groups, figsize=(fig_width, 6), squeeze=False)
        axes = axes.flatten()
        
        for i, (group_name, values) in enumerate(self.data_map.items()):
            ax = axes[i]
            values_clean = values.dropna()
            
            if len(values_clean) < 2:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
                ax.set_title(group_name, fontweight='bold')
                continue
            
            ax.hist(values_clean, bins=30, color=colors[i], alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
            
            # Add statistics text
            stats_text = (f'n = {len(values_clean)}\n'
                         f'μ = {values_clean.mean():.2f}\n'
                         f'σ = {values_clean.std():.2f}\n'
                         f'median = {values_clean.median():.2f}')
            
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                   fontsize=9, family='monospace')
            
            ax.set_xlabel('Expression', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title(group_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_groups, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Embed side-by-side
        canvas_side = FigureCanvasTkAgg(fig_side, master=self.sidebyside_tab)
        canvas_side.draw()
        
        toolbar_side = NavigationToolbar2Tk(canvas_side, self.sidebyside_tab)
        toolbar_side.update()
        toolbar_side.pack(side=tk.TOP, fill=tk.X)
        
        canvas_side.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.current_figures['sidebyside'] = fig_side
        
        # Calculate statistics
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate pairwise statistical tests - COMPLETE."""
        # Clear existing
        self.stats_tree.delete(*self.stats_tree.get_children())
        
        if len(self.data_map) < 2:
            self.stats_tree.insert('', tk.END, values=('N/A', 'N/A', 'Need at least 2 groups', '', '', ''))
            return
        
        groups = list(self.data_map.keys())
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1 = groups[i]
                group2 = groups[j]
                
                data1 = self.data_map[group1].dropna()
                data2 = self.data_map[group2].dropna()
                
                if len(data1) < 3 or len(data2) < 3:
                    self.stats_tree.insert('', tk.END, 
                                         values=(group1, group2, 'Insufficient data', '', '', ''))
                    continue
                
                # Wilcoxon rank-sum test
                try:
                    stat_w, pval_w = ranksums(data1, data2)
                except:
                    stat_w, pval_w = np.nan, np.nan
                
                # Wasserstein distance
                try:
                    wass_dist = wasserstein_distance(data1, data2)
                except:
                    wass_dist = np.nan
                
                # Cohen's d effect size
                try:
                    mean1, mean2 = data1.mean(), data2.mean()
                    std1, std2 = data1.std(), data2.std()
                    n1, n2 = len(data1), len(data2)
                    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan
                except:
                    cohens_d = np.nan
                
                # Add Wilcoxon test
                self.stats_tree.insert('', tk.END,
                                     values=(group1, group2, 'Wilcoxon', 
                                           f'{stat_w:.3f}' if not np.isnan(stat_w) else 'N/A',
                                           f'{pval_w:.4e}' if not np.isnan(pval_w) else 'N/A',
                                           f"{cohens_d:.3f}" if not np.isnan(cohens_d) else 'N/A'))
                
                # Add Wasserstein distance
                self.stats_tree.insert('', tk.END,
                                     values=(group1, group2, 'Wasserstein',
                                           f'{wass_dist:.3f}' if not np.isnan(wass_dist) else 'N/A',
                                           'N/A', 'N/A'))
    
    def _update_grouping(self):
        """Update grouping column."""
        new_grouping = self.grouping_var.get()
        
        if new_grouping not in self.df_full.columns:
            return
        
        self.grouping_col = new_grouping
        
        # Rebuild data_map based on new grouping
        self.data_map.clear()
        
        for group in self.df_full[self.grouping_col].unique():
            group_data = self.df_full[self.df_full[self.grouping_col] == group]
            if 'Gene Expression Value' in group_data.columns:
                self.data_map[str(group)] = group_data['Gene Expression Value']
        
        self._plot_distributions()
    
    def _toggle_rugs(self):
        """Toggle rug plots."""
        self.show_rugs = self.rug_var.get()
        self._plot_distributions()
    
    def _toggle_kde(self):
        """Toggle KDE overlays."""
        self._plot_distributions()
    
    def _save_plot(self):
        """Save current plot."""
        current_tab = self.notebook.index(self.notebook.select())
        
        if current_tab == 0:
            fig_key = 'overlay'
        elif current_tab == 1:
            fig_key = 'sidebyside'
        else:
            messagebox.showinfo("No Plot", "Switch to a plot tab to save.", parent=self)
            return
        
        if fig_key not in self.current_figures:
            messagebox.showwarning("No Plot", "No plot available to save.", parent=self)
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg")
            ],
            initialfile=f"comparison_{fig_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if filepath:
            try:
                self.current_figures[fig_key].savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to:\n{filepath}", parent=self)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save:\n{e}", parent=self)

    def _export_stats(self):
            """Export statistical test results."""
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"comparison_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if not filepath:
                return
            
            try:
                # Extract data from treeview
                rows = []
                for item in self.stats_tree.get_children():
                    rows.append(self.stats_tree.item(item)['values'])
                
                if not rows:
                    messagebox.showwarning("No Data", "No statistics to export.", parent=self)
                    return
                
                # Create DataFrame
                df_stats = pd.DataFrame(rows, columns=['Group 1', 'Group 2', 'Test', 'Statistic', 'P-value', 'Effect Size'])
                
                # Save
                df_stats.to_csv(filepath, index=False)
                
                messagebox.showinfo("Success", f"Statistics exported to:\n{filepath}", parent=self)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export:\n{e}", parent=self)
    
    def destroy(self):
        """Clean up before closing."""
        # Close all figures
        for fig in self.current_figures.values():
            try:
                plt.close(fig)
            except:
                pass
        
        super().destroy()
