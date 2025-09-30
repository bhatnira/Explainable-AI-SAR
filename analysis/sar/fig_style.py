import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


def apply_nature_style():
    # Base params approximating Nature style
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7.5,  # ~7–9 pt typical
        'axes.titlesize': 8,
        'axes.labelsize': 7.5,
        'axes.labelpad': 2.0,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'lines.linewidth': 0.9,
        'patch.linewidth': 0.6,
        'legend.frameon': False,
        'savefig.transparent': False,
        'figure.dpi': 100,  # keep low for canvas; control export DPI in save
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01,
        # Embed fonts for publishers
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


def colsize(single=True):
    # Nature single column ~89 mm (3.5") width, double ~183 mm (7.2"). Return (w, h)
    if single:
        return (3.5, 2.4)
    return (7.2, 3.2)


def save_figure(fig: plt.Figure, out_path: Path, dpi: int = 1200, formats=("tiff", "png")):
    out_path = Path(out_path)
    base = out_path.with_suffix("")
    for ext in formats:
        fig.savefig(base.with_suffix(f".{ext}"), dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
