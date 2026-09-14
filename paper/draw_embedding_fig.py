"""Figure: attribute document vectors and a query vector in embedding space.

The space is four-dimensional; the figure draws the first three. The three
single-attribute documents have mutually orthogonal visible parts and put the
rest of their unit length into a fourth component that is not drawn: red and
cotton have visible length sqrt(2/3), zipper only 1/sqrt(3). The query is the
unit-normalized sum of its attributes' exact linear probes, each probe being the
attribute's visible unit direction divided by its document's visible length, so the
query leans toward zipper and has no fourth component. Its cosine with x_red
and with x_zipper is the same, sqrt(2)/3 = 1/V with V = 3/sqrt(2). The
positive document x_{red,zipper}, which the probes force to be the visible sum
of x_red and x_zipper with no fourth component, scores 2/V. Dotted
perpendiculars drop the red and cotton tips onto the x and y axes, and dotted
drops from the query tip and the positive's tip to the xy plane land on the
red vector.
The visible parts are rotated 45 degrees clockwise about z while the axis
guides and the camera stay put, so the vectors sit off the coordinate axes.

Usage:  python draw_embedding_fig.py
  -> paper/figs/embedding_fig.{pdf,png}
"""

import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.proj3d import proj_transform

# --- vectors ---------------------------------------------------------------
# Visible (first three) components only. The unit directions are the standard
# basis turned 45 degrees clockwise about z; each document's visible part is
# that direction scaled to its visible length, the rest of its unit norm being
# the fourth component that is not drawn. The query is the normalized sum of
# the probes E_RED / |RED| and E_ZIPPER / |ZIPPER| and has no fourth component.
R2 = 1 / np.sqrt(2)
E_RED = np.array([R2, -R2, 0.0])
E_COTTON = np.array([R2, R2, 0.0])
E_ZIPPER = np.array([0.0, 0.0, 1.0])
RED = np.sqrt(2 / 3) * E_RED         # (1/sqrt3, -1/sqrt3, 0 | 1/sqrt3)
COTTON = np.sqrt(2 / 3) * E_COTTON   # (1/sqrt3, 1/sqrt3, 0 | 1/sqrt3)
ZIPPER = (1 / np.sqrt(3)) * E_ZIPPER # (0, 0, 1/sqrt3 | sqrt2/sqrt3)
PROBE_SUM = E_RED / np.linalg.norm(RED) + E_ZIPPER / np.linalg.norm(ZIPPER)
QUERY = PROBE_SUM / np.linalg.norm(PROBE_SUM)  # (1/sqrt6, -1/sqrt6, sqrt2/sqrt3 | 0)
POSITIVE = RED + ZIPPER                        # (1/sqrt3, -1/sqrt3, 1/sqrt3 | 0), unit norm

# --- geometry --------------------------------------------------------------
ELEV, AZIM = 20.0, 32.0   # camera
LIM = 1.25                # equal on all three axes, else the projection shears
AXIS_LEN = (1.2, 1.2, 0.95)  # how far the x, y, z axis guides run; z is short so it clears the query label
FIGSIZE = (7.0, 3.9)
CROP_PAD = 4              # points of whitespace kept around the content

# --- style -----------------------------------------------------------------
DOC = "#8c2b34"        # the three attribute vectors and their labels
ACCENT = "#2a78d6"     # the query vector
POSITIVE_COLOR = "#1b6b3a"  # the positive document and its label
CONSTRUCT = "#9fbfe6"  # the query's construction path
AXIS = "#dfe3e8"       # solid axis guides

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
})

# --- label placement -------------------------------------------------------
# Each label is placed by its CENTRE, as (x, y) in axes fractions: (0, 0) is
# the bottom-left corner of the plot box, (1, 1) the top-right. Those two
# numbers are the only thing controlling where a label sits -- move them and
# the whole label moves with them.
LABELS = (
    {"xy": (0.186, 0.165),
     "text": r"$f(x_{\mathrm{red}}) = (\frac{1}{\sqrt{3}}, -\frac{1}{\sqrt{3}}, 0, \frac{1}{\sqrt{3}})$",
     "color": DOC},
    {"xy": (0.77, 0.14),
     "text": r"$f(x_{\mathrm{cotton}}) = (\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, 0, \frac{1}{\sqrt{3}})$",
     "color": DOC},
    {"xy": (0.66, 0.60),
     "text": r"$f(x_{\mathrm{zipper}}) = (0, 0, \frac{1}{\sqrt{3}}, \frac{\sqrt{2}}{\sqrt{3}})$",
     "color": DOC},
    {"xy": (0.36, 0.85),
     "text": r"$f(q_{\mathrm{red,zipper}}) = (\frac{1}{\sqrt{6}}, -\frac{1}{\sqrt{6}}, \frac{\sqrt{2}}{\sqrt{3}}, 0)$",
     "color": ACCENT},
    {"xy": (-0.12, 0.64),
     "text": r"$f(x_{\mathrm{red,zipper}}) = (\frac{1}{\sqrt{3}}, -\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, 0)$",
     "color": POSITIVE_COLOR},
)

ORIGIN = np.zeros(3)


class Arrow3D(FancyArrowPatch):
    """An arrow that projects through the 3D transform.

    Unlike mplot3d's quiver, whose line-segment head shrinks with perspective,
    this keeps a constant, properly shaped arrowhead.
    """

    def __init__(self, xs, ys, zs, **kwargs):
        super().__init__((0, 0), (0, 0), **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = proj_transform(*self._verts3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))


def arrow(ax, vec, color, lw=1.8, zorder=5):
    """Draw a vector from the origin."""
    return ax.add_artist(Arrow3D(
        *zip(ORIGIN, vec),
        mutation_scale=12, lw=lw, arrowstyle="-|>",
        color=color, zorder=zorder, shrinkA=0, shrinkB=0,
    ))


def axis_guides(ax, lengths=AXIS_LEN):
    """Faint solid lines along the coordinate axes."""
    return [ax.plot(*zip(ORIGIN, length * axis), color=AXIS, lw=0.9,
                    zorder=0)[0] for axis, length in zip(np.eye(3), lengths)]


def dotted(ax, start, end, color=CONSTRUCT):
    """Dotted construction segment."""
    return ax.plot(*zip(start, end), color=color, lw=1.0, ls=(0, (1, 2)),
                   zorder=2)


def ink_bbox(fig, pad=CROP_PAD):
    """Display-space box around every non-background pixel of a drawn figure.

    savefig's bbox_inches="tight" crops the labels away here, because mplot3d
    leaves Text3D out of the tight bbox -- and asking a Text3D for its own
    extent does not help either, since it restores its 3D coordinates after
    drawing and then reports them through the 2D transform. Measuring the
    rendered pixels sidesteps both and catches everything actually drawn.
    """
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    rows, cols = np.nonzero(np.any(rgb < 250, axis=-1))
    height = rgb.shape[0]  # buffer rows run top-down, display coords bottom-up
    return Bbox([[cols.min(), height - rows.max()],
                 [cols.max(), height - rows.min()]]).padded(pad)


def main():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    axis_guides(ax)
    dotted(ax, RED, RED * [1, 0, 0])         # red tip onto the x axis
    dotted(ax, COTTON, COTTON * [0, 1, 0])   # cotton tip onto the y axis
    dotted(ax, COTTON, COTTON * [1, 0, 0])   # cotton tip onto the x axis
    dotted(ax, QUERY, QUERY * [1, 1, 0])     # query tip onto the xy plane
    dotted(ax, POSITIVE, POSITIVE * [1, 1, 0])  # positive tip onto the xy plane, at the red tip
    for vec in (RED, COTTON, ZIPPER):
        arrow(ax, vec, DOC)
    arrow(ax, POSITIVE, POSITIVE_COLOR)
    arrow(ax, QUERY, ACCENT, lw=2.2, zorder=6)

    for label in LABELS:
        ax.text2D(*label["xy"], label["text"], transform=ax.transAxes,
                  color=label["color"], ha="center", va="center")

    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(0, LIM)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()  # no panes, grid, or ticks

    # The axes keep the width of a 5.2in canvas; the extra 1.9in on the left is room for
    # labels placed at negative axes x, which the ink crop then trims to.
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.06, top=1.0)
    crop = ink_bbox(fig).transformed(fig.dpi_scale_trans.inverted())

    out_dir = pathlib.Path(__file__).resolve().parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"embedding_fig{suffix}"
        fig.savefig(path, dpi=300, bbox_inches=crop,
                    facecolor="white", transparent=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
