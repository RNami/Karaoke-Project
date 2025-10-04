import tkinter as tk
from tkinter import ttk

# ----------------------------
# Global defaults
# ----------------------------
DEFAULT_PADX = 5
DEFAULT_PADY = 5
DEFAULT_STICKY = "w"


# ----------------------------
# Internal helper
# ----------------------------
def _grid_widget(widget, row=None, column=None, **grid_opts):
    """Internal helper to grid a widget with defaults."""
    if row is not None and column is not None:
        widget.grid(
            row=row,
            column=column,
            padx=grid_opts.pop("padx", DEFAULT_PADX),
            pady=grid_opts.pop("pady", DEFAULT_PADY),
            sticky=grid_opts.pop("sticky", DEFAULT_STICKY),
            columnspan=grid_opts.pop("columnspan", 1),
        )
    return widget


# ----------------------------
# Basic widget creators
# ----------------------------
def make_label(parent, text=None, textvariable=None, row=None, column=None, **kwargs):
    """Create a label with default padding and sticky."""
    lbl = ttk.Label(parent, text=text, textvariable=textvariable)
    return _grid_widget(lbl, row, column, **kwargs)


def make_entry(parent, textvariable=None, width=10, row=None, column=None, **kwargs):
    """Create an entry with default padding and sticky."""
    ent = ttk.Entry(parent, textvariable=textvariable, width=width)
    return _grid_widget(ent, row, column, **kwargs)


def make_button(parent, text, command=None, row=None, column=None, **kwargs):
    """Create a button with default padding and sticky."""
    btn = ttk.Button(parent, text=text, command=command)
    return _grid_widget(btn, row, column, **kwargs)


def make_combobox(parent, values, textvariable=None, state="readonly",
                  width=30, row=None, column=None, **kwargs):
    """Create a combobox with default padding and sticky."""
    combo = ttk.Combobox(parent, textvariable=textvariable,
                         values=values, state=state, width=width)
    return _grid_widget(combo, row, column, **kwargs)


def make_progressbar(parent, variable, maximum=100, length=200,
                     row=None, column=None, **kwargs):
    """Create a progressbar with default padding and sticky."""
    bar = ttk.Progressbar(parent, variable=variable,
                          maximum=maximum, length=length)
    return _grid_widget(bar, row, column, **kwargs)


def make_textbox(parent, height=8, width=50, row=None, column=None, **kwargs):
    """Create a text box with default padding and sticky."""
    box = tk.Text(parent, height=height, width=width)
    return _grid_widget(box, row, column, **kwargs)


# ----------------------------
# Containers
# ----------------------------
def make_frame(parent, relief="flat", borderwidth=0, row=None, column=None, **kwargs):
    """Create a frame with default grid layout."""
    frm = ttk.Frame(parent, relief=relief, borderwidth=borderwidth)
    return _grid_widget(frm, row, column, **kwargs)

def make_labelframe(parent, text="", relief="groove", borderwidth=2,
                    row=None, column=None, **kwargs):
    """Create a labeled frame with default grid layout."""
    lfrm = ttk.LabelFrame(parent, text=text, relief=relief, borderwidth=borderwidth)
    return _grid_widget(lfrm, row, column, **kwargs)


# ----------------------------
# Grid configuration utilities
# ----------------------------
def configure_grid(frame, rows=None, cols=None, weight=1):
    """Configure grid responsiveness; accepts int or list."""
    if rows is not None:
        if isinstance(rows, int):
            rows = [rows]
        for r in rows:
            frame.rowconfigure(r, weight=weight)

    if cols is not None:
        if isinstance(cols, int):
            cols = [cols]
        for c in cols:
            frame.columnconfigure(c, weight=weight)

def expand(widget):
    """
    Shortcut to make a widget expand in all directions (sticky 'nsew').
    Example:
        expand(my_frame)
    """
    widget.grid_configure(sticky="nsew")
    return widget