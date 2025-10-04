import tkinter as tk
from tkinter import ttk

DEFAULT_PADX = 5
DEFAULT_PADY = 5
DEFAULT_STICKY = "w"


def make_label(parent, text=None, textvariable=None, row=None, column=None, **grid_opts):
    """Create a label with default padding and sticky."""
    lbl = ttk.Label(parent, text=text, textvariable=textvariable)
    if row is not None and column is not None:
        lbl.grid(row=row, column=column,
                 padx=DEFAULT_PADX, pady=DEFAULT_PADY,
                 sticky=grid_opts.get("sticky", DEFAULT_STICKY),
                 columnspan=grid_opts.get("columnspan", 1))
    return lbl

def make_entry(parent, textvariable=None, width=10, row=None, column=None, **grid_opts):
    """Create an entry with default padding and sticky."""
    ent = ttk.Entry(parent, textvariable=textvariable, width=width)
    if row is not None and column is not None:
        ent.grid(row=row, column=column,
                 padx=DEFAULT_PADX, pady=DEFAULT_PADY,
                 sticky=grid_opts.get("sticky", DEFAULT_STICKY))
    return ent


def make_button(parent, text, command=None, row=None, column=None, **grid_opts):
    """Create a button with default padding and sticky."""
    btn = ttk.Button(parent, text=text, command=command)
    if row is not None and column is not None:
        btn.grid(row=row, column=column,
                 padx=DEFAULT_PADX, pady=DEFAULT_PADY,
                 sticky=grid_opts.get("sticky", DEFAULT_STICKY),
                 columnspan=grid_opts.get("columnspan", 1))
    return btn


def make_combobox(parent, values, textvariable=None, state="readonly",
                  width=30, row=None, column=None, **grid_opts):
    """Create a combobox with default padding and sticky."""
    combo = ttk.Combobox(parent, textvariable=textvariable,
                         values=values, state=state, width=width)
    if row is not None and column is not None:
        combo.grid(row=row, column=column,
                   padx=DEFAULT_PADX, pady=DEFAULT_PADY,
                   sticky=grid_opts.get("sticky", DEFAULT_STICKY))
    return combo


def make_progressbar(parent, variable, maximum=100, length=200,
                     row=None, column=None, **grid_opts):
    """Create a progressbar with default padding and sticky."""
    bar = ttk.Progressbar(parent, variable=variable,
                          maximum=maximum, length=length)
    if row is not None and column is not None:
        bar.grid(row=row, column=column,
                 padx=DEFAULT_PADX, pady=DEFAULT_PADY,
                 sticky=grid_opts.get("sticky", DEFAULT_STICKY))
    return bar


def make_textbox(parent, height=8, width=50, row=None, column=None, **grid_opts):
    """Create a text box with default padding and sticky."""
    box = tk.Text(parent, height=height, width=width)
    if row is not None and column is not None:
        box.grid(row=row, column=column,
                 padx=DEFAULT_PADX, pady=DEFAULT_PADY,
                 sticky=grid_opts.get("sticky", DEFAULT_STICKY),
                 columnspan=grid_opts.get("columnspan", 1))
    return box
