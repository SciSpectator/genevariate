"""
Dialog windows for GeneVariate.
Provides simple dialog boxes for user interaction.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path


class SavePlotsDialog(tk.Toplevel):
    """Dialog for saving plots with options."""
    
    def __init__(self, parent, default_prefix="plot"):
        super().__init__(parent)
        self.title("Save Plot")
        self.geometry("400x300")
        self.transient(parent)
        self.grab_set()
        
        self.result = None
        
        ttk.Label(self, text="Save Plot Options", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        # Filename
        ttk.Label(self, text="Filename prefix:").pack(anchor=tk.W, padx=20)
        self.filename_entry = ttk.Entry(self, width=40)
        self.filename_entry.insert(0, default_prefix)
        self.filename_entry.pack(padx=20, pady=5)
        
        # Format
        ttk.Label(self, text="Format:").pack(anchor=tk.W, padx=20, pady=(10, 0))
        self.format_var = tk.StringVar(value="png")
        
        format_frame = ttk.Frame(self)
        format_frame.pack(padx=20, pady=5)
        
        for fmt in ['png', 'pdf', 'svg', 'jpg']:
            ttk.Radiobutton(format_frame, text=fmt.upper(), variable=self.format_var, value=fmt).pack(side=tk.LEFT, padx=5)
        
        # DPI
        ttk.Label(self, text="DPI (resolution):").pack(anchor=tk.W, padx=20, pady=(10, 0))
        self.dpi_var = tk.StringVar(value="300")
        
        dpi_frame = ttk.Frame(self)
        dpi_frame.pack(padx=20, pady=5)
        
        for dpi in ['150', '300', '600']:
            ttk.Radiobutton(dpi_frame, text=dpi, variable=self.dpi_var, value=dpi).pack(side=tk.LEFT, padx=5)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=5)
    
    def save(self):
        """Save with selected options."""
        self.result = {
            'filename': self.filename_entry.get(),
            'format': self.format_var.get(),
            'dpi': int(self.dpi_var.get())
        }
        self.destroy()
    
    def cancel(self):
        """Cancel save."""
        self.result = None
        self.destroy()


class SubsetDisplayOptionsDialog(tk.Toplevel):
    """Dialog for display options."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Display Options")
        self.geometry("350x200")
        self.transient(parent)
        self.grab_set()
        
        self.result = None
        
        ttk.Label(self, text="Display Options", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        # Show legend
        self.legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Show legend", variable=self.legend_var).pack(anchor=tk.W, padx=20, pady=5)
        
        # Show grid
        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Show grid", variable=self.grid_var).pack(anchor=tk.W, padx=20, pady=5)
        
        # Point size
        ttk.Label(self, text="Point size:").pack(anchor=tk.W, padx=20, pady=(10, 0))
        self.size_var = tk.IntVar(value=50)
        ttk.Scale(self, from_=10, to=200, variable=self.size_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Apply", command=self.apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=5)
    
    def apply(self):
        """Apply options."""
        self.result = {
            'legend': self.legend_var.get(),
            'grid': self.grid_var.get(),
            'point_size': self.size_var.get()
        }
        self.destroy()
    
    def cancel(self):
        """Cancel."""
        self.result = None
        self.destroy()


class SelectColumnsDialog(tk.Toplevel):
    """Dialog for selecting columns."""
    
    def __init__(self, parent, columns):
        super().__init__(parent)
        self.title("Select Columns")
        self.geometry("400x500")
        self.transient(parent)
        self.grab_set()
        
        self.result = None
        
        ttk.Label(self, text="Select columns to include:", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        # Listbox with checkboxes
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=20)
        scrollbar = ttk.Scrollbar(list_frame, command=self.listbox.yview)
        self.listbox.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        for col in columns:
            self.listbox.insert(tk.END, col)
        
        # Select all by default
        self.listbox.select_set(0, tk.END)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Select All", command=lambda: self.listbox.select_set(0, tk.END)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear All", command=lambda: self.listbox.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="OK", command=self.ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=5)
    
    def ok(self):
        """OK clicked."""
        selected_indices = self.listbox.curselection()
        self.result = [self.listbox.get(i) for i in selected_indices]
        self.destroy()
    
    def cancel(self):
        """Cancel clicked."""
        self.result = None
        self.destroy()
