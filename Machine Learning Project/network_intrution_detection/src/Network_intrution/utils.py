# Updated utility.py
# Supports Matplotlib, Seaborn (via Matplotlib), Plotly and flexible saving
# - Avoids use of unreliable Plotly "get_current_figure" API
# - Allows explicit plotly figure lists to be saved
# - Better detection of Matplotlib Axes / Figures
# - Returns list of saved Path objects

import re
import warnings
from pathlib import Path
from typing import Optional, Union, List, Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure
from matplotlib.axes import Axes as MplAxes

# Optional Plotly imports
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _HAS_PLOTLY = True
except Exception:
    go = None  # type: ignore
    pio = None  # type: ignore
    _HAS_PLOTLY = False

# Try to enable static image export (kaleido)
try:
    from plotly.io._kaleido import scope as kaleido_scope  # type: ignore
    _HAS_KALEIDO = True
except Exception:
    _HAS_KALEIDO = False

DEFAULT_OUTPUT_FOLDER = Path("figures")


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[_\s]+', '_', filename)
    filename = filename.strip(' _')
    if len(filename) > max_length:
        filename = filename[:max_length]
    return filename


def _ensure_output_folder(folder: Union[str, Path]) -> Path:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_matplotlib_title(fig: MplFigure) -> Optional[str]:
    """Extract a sensible title from a Matplotlib figure."""
    if fig is None:
        return None
    # Figure suptitle
    if getattr(fig, '_suptitle', None) is not None:
        try:
            t = fig._suptitle.get_text()
            if t and t.strip():
                return t.strip()
        except Exception:
            pass

    # Per-axis title
    try:
        for ax in fig.axes:
            t = ax.get_title()
            if t and t.strip():
                return t.strip()
    except Exception:
        pass

    # Fallback to axis labels
    try:
        for ax in fig.axes:
            for lbl in (ax.get_xlabel(), ax.get_ylabel()):
                if lbl and lbl.strip():
                    return lbl.strip()
    except Exception:
        pass

    return None


def get_plotly_title(fig) -> Optional[str]:
    """Extract title from a Plotly figure if available."""
    if not _HAS_PLOTLY or fig is None:
        return None
    try:
        if hasattr(fig, 'layout') and getattr(fig.layout, 'title', None) is not None:
            t = fig.layout.title.text
            if t and str(t).strip():
                return str(t).strip()
    except Exception:
        pass
    return None


def save_plotly(
    fig,
    filename: Optional[str] = None,
    prefix: str = '',
    suffix: str = '',
    counter: Optional[int] = None,
    output_folder: Union[str, Path] = DEFAULT_OUTPUT_FOLDER,
    html: bool = True,
    static_format: str = 'png',
    static_dpi: int = 300,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None,
    verbose: bool = True,
) -> List[Path]:
    """Save a Plotly figure to HTML and optionally to a static image.

    Note: this function expects a plotly.graph_objects.Figure (or compatible)
    to be passed explicitly.
    """
    if not _HAS_PLOTLY:
        raise ImportError("Plotly is not available. Install with: pip install plotly")

    out = _ensure_output_folder(output_folder)

    title = get_plotly_title(fig) or "untitled_plotly_chart"
    base = sanitize_filename(title)
    if filename:
        base = sanitize_filename(filename)
    if prefix:
        base = f"{prefix}_{base}"
    if suffix:
        base = f"{base}_{suffix}"
    if counter is not None:
        base = f"{base}_{counter:03d}"

    saved: List[Path] = []

    if html:
        html_path = out / f"{base}.html"
        fig.write_html(str(html_path), include_plotlyjs='cdn', config={'responsive': True})
        saved.append(html_path)
        if verbose:
            print(f"✅ Plotly HTML saved: {html_path}")

    if static_format and _HAS_KALEIDO:
        img_path = out / f"{base}.{static_format.lstrip('.') }"
        # scale default: scale such that DPI ~ static_dpi (Plotly default 96)
        s = scale or (static_dpi / 96.0)
        fig.write_image(str(img_path), format=static_format, width=width, height=height, scale=s)
        saved.append(img_path)
        if verbose:
            size_info = f"{width or 'auto'}x{height or 'auto'}"
            print(f"✅ Plotly static {static_format.upper()} saved: {img_path} ({size_info}, {static_dpi} DPI)")
    elif static_format and verbose and not _HAS_KALEIDO:
        print("⚠️  Kaleido not available. Static image export skipped. Install with: pip install -U kaleido")

    return saved


def save_chart(
    fig: Optional[Union[MplFigure, MplAxes, object]] = None,
    *,
    show: bool = True,   # ✅ ADD THIS
    dpi: int = 300,
    output_folder: Union[str, Path] = DEFAULT_OUTPUT_FOLDER,
    format: str = 'png',
    bbox_inches: str = 'tight',
    pad_inches: float = 0.1,
    facecolor: Optional[str] = None,
    edgecolor: Optional[str] = None,
    transparent: bool = False,
    close_figure: bool = False,
    verbose: bool = True,
    filename: Optional[str] = None,
    prefix: str = '',
    suffix: str = '',
    counter: Optional[int] = None,
    # Plotly-specific passthroughs
    plotly_html: bool = True,
    plotly_static: bool = True,
    plotly_static_format: str = 'png',
    plotly_width: Optional[int] = None,
    plotly_height: Optional[int] = None,
    plotly_scale: Optional[float] = None,
) -> List[Path]:
    """Save a Matplotlib (or Plotly) figure.

    If `fig` is None, the currently active Matplotlib figure (plt.gcf()) is used.
    If `fig` is a Plotly figure, it will be saved via save_plotly().
    If `fig` is a Matplotlib Axes, its parent figure will be used.

    Returns list of saved Path objects.
    """
    out = _ensure_output_folder(output_folder)
    saved_paths: List[Path] = []

    # If no explicit fig provided, fallback to current Matplotlib figure
    if fig is None:
        fig = plt.gcf()

    # If user passed an Axes, extract the parent Figure
    if isinstance(fig, MplAxes):
        fig = fig.figure

    # Detect Plotly figure
    if _HAS_PLOTLY and (go is not None) and isinstance(fig, go.Figure):
        # Map plotly options
        pf_format = plotly_static_format if plotly_static else None
        saved = save_plotly(
            fig=fig,
            filename=filename,
            prefix=prefix,
            suffix=suffix,
            counter=counter,
            output_folder=out,
            html=plotly_html,
            static_format=pf_format,
            static_dpi=dpi,
            width=plotly_width,
            height=plotly_height,
            scale=plotly_scale,
            verbose=verbose,
        )
        saved_paths.extend(saved)
        return saved_paths

    # Otherwise assume Matplotlib figure-like object
    if hasattr(fig, 'savefig'):
        # Construct a filename
        if filename is None:
            title = get_matplotlib_title(fig)
            base = sanitize_filename(title) if title else 'untitled_chart'
            if prefix:
                base = f"{prefix}_{base}"
            if suffix:
                base = f"{base}_{suffix}"
            if counter is not None:
                base = f"{base}_{counter:03d}"
        else:
            base = sanitize_filename(filename)

        full_name = f"{base}.{format.lstrip('.')}"
        file_path = Path(out) / full_name

        if file_path.exists() and verbose:
            warnings.warn(f"File {file_path} already exists. Overwriting.", UserWarning)

        save_kwargs = {
            'dpi': dpi,
            'bbox_inches': bbox_inches,
            'pad_inches': pad_inches,
            'format': format,
            'transparent': transparent,
        }
        if facecolor:
            save_kwargs['facecolor'] = facecolor
        if edgecolor:
            save_kwargs['edgecolor'] = edgecolor

        # Use Matplotlib's savefig
        fig.savefig(str(file_path), **save_kwargs)
        saved_paths.append(file_path)

        if verbose:
            try:
                size = fig.get_size_inches()
                print(f"✅ Matplotlib chart saved: {file_path} (Size: {size[0]:.1f}x{size[1]:.1f} in, DPI: {dpi})")
            except Exception:
                print(f"✅ Matplotlib chart saved: {file_path}")
            if len(getattr(fig, 'axes', [])) == 0:
                print("⚠️  Warning: The figure has 0 axes. Did you forget to add a plot?")

        if show and not close_figure:
            try:
                from IPython import get_ipython
                if get_ipython() is not None:
                    from IPython.display import display
                    display(fig)
                else:
                    plt.show(block=False)
            except Exception:
                try:
                    plt.show(block=False)
                except Exception:
                    pass



        if close_figure:
            try:
                plt.close(fig)
            except Exception:
                pass

        return saved_paths

    # If we reach here, unsupported object
    raise TypeError("Unsupported figure object passed to save_chart(). "
                    "Pass a Matplotlib Figure/Axes or a Plotly Figure.")


def save_all_figures_in_session(
    prefix: str = '',
    suffix: str = '',
    output_folder: Union[str, Path] = DEFAULT_OUTPUT_FOLDER,
    close_figures: bool = False,
    plotly_figs: Optional[Iterable[object]] = None,
    **kwargs,
) -> List[Path]:
    """Save all open Matplotlib figures and any Plotly figures the user passes in.

    Because Plotly does not maintain a global "current figure" like Matplotlib,
    pass a list/iterable of Plotly figures via `plotly_figs` if you want them saved.
    """
    out = _ensure_output_folder(output_folder)
    saved: List[Path] = []

    # Save Matplotlib figures
    import matplotlib._pylab_helpers as pylab_helpers
    manager_ids = list(pylab_helpers.Gcf.figs.keys())

    for i, manager_id in enumerate(manager_ids):
        plt.figure(manager_id)
        mpl_fig = plt.gcf()
        # delegate to save_chart for consistent naming
        paths = save_chart(
            fig=mpl_fig,
            filename=None,
            prefix=prefix,
            suffix=suffix,
            counter=i,
            output_folder=out,
            **{k: v for k, v in kwargs.items()}
        )
        saved.extend(paths)
        if close_figures:
            try:
                plt.close(mpl_fig)
            except Exception:
                pass

    # Save Plotly figures provided by the caller
    if _HAS_PLOTLY and plotly_figs is not None:
        j = len(manager_ids)
        for k, pf in enumerate(plotly_figs):
            try:
                paths = save_plotly(
                    fig=pf,
                    filename=None,
                    prefix=prefix,
                    suffix=suffix,
                    counter=j + k,
                    output_folder=out,
                    html=kwargs.get('plotly_html', True),
                    static_format=kwargs.get('plotly_static_format', 'png') if kwargs.get('plotly_static', True) else None,
                    static_dpi=kwargs.get('dpi', 300),
                    width=kwargs.get('plotly_width', None),
                    height=kwargs.get('plotly_height', None),
                    scale=kwargs.get('plotly_scale', None),
                    verbose=kwargs.get('verbose', True),
                )
                saved.extend(paths)
            except Exception as e:
                warnings.warn(f"Failed to save Plotly figure #{k}: {e}")

    total = len(saved)
    if total == 0:
        warnings.warn("No figures found to save.", UserWarning)
    else:
        print(f"✅ Saved {total} figure(s) to {out}")

    return saved


class AutoSaveFigure:
    """Context manager that automatically saves a given figure (Matplotlib or Plotly)

    Usage:
        with AutoSaveFigure(fig=plt.gcf(), output_folder='out', prefix='run'):
            # plotting code
            pass
    If no fig is provided, the current Matplotlib figure (plt.gcf()) is used.
    For Plotly, pass the figure explicitly.
    """

    def __init__(self, fig: Optional[object] = None, **save_kwargs):
        if 'close_figure' not in save_kwargs:
            save_kwargs['close_figure'] = False
        self.save_kwargs = save_kwargs
        self.provided_fig = fig
        self.fig = None

    def __enter__(self):
        if self.provided_fig is not None:
            self.fig = self.provided_fig
        else:
            # prefer Matplotlib current figure
            self.fig = plt.gcf()
        return self.fig

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fig is not None:
            try:
                # delegate to save_chart which handles both Matplotlib and Plotly
                save_chart(fig=self.fig, **self.save_kwargs)
            except Exception as e:
                warnings.warn(f"AutoSaveFigure failed to save the figure: {e}")
        return False


__all__ = [
    'save_chart', 'save_all_figures_in_session', 'AutoSaveFigure',
    'DEFAULT_OUTPUT_FOLDER', 'save_plotly', 'sanitize_filename'
]
