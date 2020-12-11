import scipy.signal
from matplotlib.lines import Line2D
from matplotlib import rc
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import NullFormatter
import matplotlib.gridspec as gridspec
import matplotlib.colors
from matplotlib import pyplot as plt
import matplotlib as mpl
from slopemarker import slope_marker
import numpy as np
import sys
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import io
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable

from palettable.colorbrewer.diverging import RdYlBu_7
from lerner_group.visualization_tools import histogram_log_bins

# colors = ['palevioletred', 'darkslateblue', 'mediumseagreen', 'mediumpurple', 'darkorange', 'firebrick', 'mediumturquoise', 'olive', 'indigo', 'goldenrod']
colors = sns.color_palette('colorblind')
markers = list(Line2D.filled_markers)
annot_font_size = 8
label_font_size = 8
text_font_size = 10
legend_font_size = label_font_size
fit_lw = 0.75
markeredgewidth = 0.5
markersize = 4

pre_width_points = 246
pre_double_width_points = 510
points_per_inch = 72.27
pre_width = pre_width_points / points_per_inch
pre_double_width = pre_double_width_points / points_per_inch


def myshow():
    plt.tight_layout()
    plt.show()

def init_fig(width=None, height=None, grid=(1,1), locs=None, colspans=None, rowspans=None, projections=None, facecolor='white', ax_width=0.5, text_font_size=text_font_size, label_font_size=label_font_size, default_size=3.5):
    # locs: 
    if not width:
        width = grid[1]*default_size
    if not height:
        height = grid[0]*default_size

    if not locs:
        ncols, nrows = grid
        locs = []
        colspans = []
        rowspans = []
        for i in range(ncols):
            for j in range(nrows):
                locs.append((i, j))
                colspans.append(1)
                rowspans.append(1)

    if not projections:
        projections = [None for i in range(grid[0]*grid[1])]

    params = {
        'text.usetex': True,
        'text.latex.preamble': "\\usepackage{physics}\n\\usepackage{amsmath}\n\\usepackage{cmbright}\n\\DeclareMathOperator{\\cdf}{CDF}\n\\DeclareMathOperator{\\pdf}{PDF}",
        'xtick.labelsize': label_font_size,
        'ytick.labelsize': label_font_size,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': ax_width,
        'ytick.major.width': ax_width,
        'axes.labelsize': text_font_size,
        'axes.linewidth': ax_width,
        'font.size': text_font_size
    }
    plt.rcParams.update(params)
    fig = plt.gcf()
    fig.set_facecolor(facecolor)
    fig.set_size_inches(width, height)

    axes = ()
    for n, loc in enumerate(locs):
        axes += (plt.subplot2grid(grid, loc, rowspan=rowspans[n], colspan=colspans[n], projection=projections[n]), )

    if grid == (1, 1):
        return axes[0]
    else:
        return axes


def nice_legend(ax, box=False, labels=None, handles=None, **kwargs):

    default_kwargs = {'ncol': 1, 'fontsize': legend_font_size, 'title': None, 'loc': 'best', 'title_fontsize': legend_font_size}
    for key in default_kwargs:
        if not key in kwargs:
            kwargs[key] = default_kwargs[key]

    handletextpad = 0.4
    borderpad = 0.4
    handlelength = 1.25
    if box:
        framealpha = 1 
        frameon = True
    else:
        frameon = False
        framealpha = 0
    scatteryoffsets = [0.5]
    if labels:
        legend = ax.legend(
            labels, frameon=frameon, handletextpad=handletextpad, borderpad=borderpad, handlelength=handlelength,
            shadow=False, framealpha=framealpha, scatteryoffsets=scatteryoffsets, **kwargs)
    else:
        if handles:
            legend = ax.legend(
                handles=handles, frameon=frameon, handletextpad=handletextpad, borderpad=borderpad, handlelength=handlelength,
                shadow=False, framealpha=framealpha, scatteryoffsets=scatteryoffsets, **kwargs)
        else:
            legend = ax.legend(
                frameon=frameon, handletextpad=handletextpad, borderpad=borderpad, handlelength=handlelength,
                shadow=False, framealpha=framealpha, scatteryoffsets=scatteryoffsets, **kwargs)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_linewidth(0.5)

    # legend.get_title().set_fontsize(kwargs['fontsize'])
    ax.add_artist(legend)
    return legend


def nice_triangle(ax, location, slope, invert=False, size_frac=0.18, font_size=annot_font_size, custom_rise_label=None, pad_frac_x=0.2, pad_frac_y=0.2):
    slope_marker(location, slope, ax=ax, invert=invert, size_frac=size_frac, custom_rise_label=custom_rise_label, fontsize=font_size,
        poly_kwargs={'facecolor': 'white', 'edgecolor':'black', 'linewidth':fit_lw}, pad_frac_x=pad_frac_x, pad_frac_y=pad_frac_y)

def autolabel_bar(rects, ax, labels, rotate=0, offset=3):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate(labels[i],
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, offset),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=rotate)


def plot_phonon_widths(ax, filename, N, marker, color):
    data = np.loadtxt(filename)
    omega, delta_omega, n_q = data[:,0], data[:,1], data[:,2]
    x = omega * np.sqrt(n_q) / np.sqrt(N)

    ax.loglog(x, delta_omega, marker=marker, color=color, ls='None', markeredgewidth=markeredgewidth, markeredgecolor='k',markersize=markersize)


def nice_plot(ax, x, y, xerr=None, yerr=None, color=colors[0], ecolor='k', marker='o', xscale='linear', yscale='linear', ls='None', ms=markersize, markeredgecolor='k', markeredgewidth=markeredgewidth, alpha=1, markerfacecolor=None, label=None, barsabove=False, lw=fit_lw, zorder=None):
    if not markerfacecolor:
        markerfacecolor = color
    if xerr is not None or yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='-', marker=marker, color=color, ecolor=ecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor, ms=ms, lw=lw, ls=ls, elinewidth=fit_lw, capsize=2, capthick=0.5, barsabove=barsabove, label=label, zorder=zorder)
    else:
        ax.plot(x, y, marker=marker, color=color, ls=ls, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor, markersize=ms, alpha=alpha, markerfacecolor=markerfacecolor, label=label, lw=lw, zorder=zorder)


def lighten_color(color, amount):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def draw_line_loglog(ax, exponent, prefactor=1, color='k', ls='-', lw=0.75):
    xlim = np.asarray(ax.get_xlim())
    ylim = np.asarray(ax.get_ylim())
    ax.loglog(xlim, prefactor*xlim**exponent, color=color, ls=ls, lw=lw)
    ax.set(xlim=xlim, ylim=ylim)


def cumulative_histogram_log(x, x_min=None, x_max=None, nbins=20):
    """
    Generate cumulative histogram with logarithmically spaced bins.
    """
    if not x_min:
        x_min = np.min(x)
    if not x_max:
        x_max = np.max(x)

    # This is the factor that each subsequent bin is larger than the next.
    # growth_factor = (x_max/x_min) ** (1/(nbins + 1))
    # Generates logarithmically spaced points from x_min to x_max.
    bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), num=nbins+1)
    # We don't need the second argument (which are again the bin edges).
    # It's conventional to denote arguments you don't intend to use with _.

    bin_counts,_ = np.histogram(x, bins=bin_edges)
    bin_counts = bin_counts.astype(float)
    cumulative = np.cumsum(bin_counts)

    right_edges = bin_edges[1:]

    return cumulative, right_edges


def nice_hist(x, x_min=None, x_max=None, bins=10, min_hits=0, density=False):
    counts, edges = np.histogram(x, bins=bins, density=density)
    nonzero_idx = np.nonzero(counts > 0)
    nonzero_counts = counts[nonzero_idx]
    min_unit = np.min(nonzero_counts)
    idx = np.nonzero( counts >= min_hits * min_unit)
    centers = (edges[:-1] + edges[1:]) / 2

    return counts[idx], centers[idx]


def add_labels(axes, style=r'(%s)', shift_x_em=0.0):
    """
    Adds figure labels (a), (b), (c), etc. to axes.
    Must be called after plt.tight_layout().
    """

    alphabet = 'abcdefghijkl'
    labels = [style % letter for letter in alphabet]

    # EXTREME hack alert.
    # For some magical reason, this is necessary to get the bounding boxes correct.
    axes[0].figure.savefig("/tmp/hallo.pdf")   

    for i, ax in enumerate(axes):
        transform = ax.transAxes
        inv_transform = ax.transAxes.inverted()
        ylabelbox = ax.yaxis.label.get_window_extent()
        fontsize = ax.yaxis.label.get_size()
        axbox = ax.get_window_extent()
        labelx_pixel = ylabelbox.x0 + shift_x_em*fontsize  # move 0.5 em to the left to align better.
        labely_pixel = axbox.y1
        labelx, labely = inv_transform.transform((labelx_pixel, labely_pixel))
        fontsize = ax.yaxis.label.get_font_properties().get_size()
        ax.text(labelx, labely, labels[i], transform=transform, va='top', ha='left', fontsize=fontsize)


# define a function which returns an image as numpy array from figure
# def fig_asarray(fig=plt.gcf(), dpi=180):
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=dpi)
#     buf.seek(0)
#     img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#     buf.close()
#     img = cv2.imdecode(img_arr, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     return img
def match_slope_to_figure_angle(ax, slope):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    # x_sz, y_sz = plt.gcf().get_size_inches()
    bbox = ax.get_window_extent()
    x_sz, y_sz = bbox.width, bbox.height
    x_factor = x_sz / (np.log10(x_max) - np.log10(x_min))
    y_factor = y_sz / (np.log10(y_max) - np.log10(y_min))
    adjusted_slope = (slope * y_factor / x_factor)
    return (360/(2*np.pi)) * np.arctan2(adjusted_slope, 1)

def nice_colorbar(ax, cmap=None, norm=None, **kwargs):
    defaults = dict(size="5%", pad=0.05, pack_start=True)
    for key in defaults:
        if not key in kwargs:
            kwargs[key] = defaults[key]

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(**kwargs)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb, orientation='horizontal')
    return ax_cb
