import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import subprocess
from io import StringIO
import os

from lerner_group.visualization_tools import histogram_log_bins

OTHER = 0; FCC = 1; HCP = 2; BCC = 3; ICO = 4;  # ovito crystal analysis codes.
OVITO_CRYSTAL_CODES = [OTHER, FCC, HCP, BCC, ICO]

CRYSTAL_CODE_NAMES = {
    OTHER: 'other',
    FCC:   'fcc',
    HCP:   'hcp',
    BCC:   'bcc',
    ICO:   'ico'
}


def logbin(x, x_min=None, x_max=None, n_bins=100):
    """
    Generate histogram with logarithmically spaced bins.
    """
    if not x_max:
        x_max = np.max(x)
    if not x_min:
        x_min = np.min(x)
    data_size = x.size
    # This is the factor that each subsequent bin is larger than the next.
    growth_factor = (x_max/x_min) ** (1/(n_bins + 1))
    # Generates logarithmically spaced points from x_min to x_max.
    bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), num=n_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    # We don't need the second argument (which are again the bin edges).
    # It's conventional to denote arguments you don't intend to use with _.
    bin_counts, _ = np.histogram(x, bins=bin_edges, density=True)
    bin_counts = bin_counts.astype(np.float)
    # bin_counts /= data_size
    # Rescale bin counts by their relative sizes.
    # for bin_index in range(np.size(bin_counts)):
    #     bin_counts[bin_index] = bin_counts[bin_index] / (growth_factor**bin_index)
    
    non_empty_bins = np.nonzero(bin_counts)
    return bin_counts[non_empty_bins], bin_centers[non_empty_bins]


def running_avg_log(x, y, x_min, x_max, num_of_bins, method=np.mean):
    """
    Running average with logarithmically spaced bins.
    """
    # Get non-empty bin_edges from logbin (don't need the counts).
    _, bin_edges = logbin(x, x_min, x_max, num_of_bins)
    np.append(bin_edges, x_max)

    # We have one less bin than bin edges.
    running_avg = np.zeros(bin_edges.shape[0] - 1)
    for i in range(bin_edges.shape[0] - 1):
        left_edge = bin_edges[i]
        right_edge = bin_edges[i + 1]
        this_bin = y[np.nonzero( np.logical_and(x > left_edge, x < right_edge) )]
        running_avg[i] = method(this_bin)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return running_avg, bin_centers

    
def fit_power_law(x, y, exponent=None):
    if exponent:
        def f(logx, log_prefactor):
            logy = log_prefactor + exponent*logx
            return logy
    else:
        def f(logx, exponent, log_prefactor):
            logy = log_prefactor + exponent*logx
            return logy
    
    logx = np.log(x)
    logy = np.log(y)
    popt, _ = curve_fit(f, logx, logy)

    if exponent:
        log_prefactor = popt[0]
        prefactor = np.exp(log_prefactor)
        return prefactor
    else:
        exponent = popt[0]
        log_prefactor = popt[1]
        prefactor = np.exp(log_prefactor)
        return exponent, prefactor


def power_law_random_variate(y, n,  x0, x1):
    """
    Transforms uniform [0, 1] random number y into x distributed as C x^n on [x0, x1]. 
    """

    return ( y * ( x1**(n+1) - x0**(n+1) ) + x0**(n+1) ) ** (1 / (n+1))

# Thanks Ismani.
def qq_plotter(ax, samples1, samples2, nquantiles=100,
               edgecolors=None, linecolor='k', marker='o'):
    """
    Plots quantiles of empirical distribution functions (linearly interpolating
    between data points)
    """
    # Plot line y = x
    both = np.concatenate((samples1, samples2))
    both_min = np.min(both)
    both_max = np.max(both)
    #  both_len = both_max - both_min
    #  tweak = both_len / 10

    #  domain = np.linspace(np.min(both) - tweak, np.max(both) + tweak)
    domain = np.linspace(np.min(both), np.max(both))

    ax.plot(domain, domain, '-', linewidth=0.5, color=linecolor, zorder=1)

    # Plot quantiles
    quantile_space = np.linspace(0, 1, nquantiles)

    quantiles1 = np.quantile(samples1, quantile_space)
    quantiles2 = np.quantile(samples2, quantile_space)

    #  ax.scatter(quantiles1, quantiles2, marker=marker)
    if edgecolors is None:
        edgecolors = np.random.choice(colors)

    ax.scatter(quantiles1, quantiles2, marker=marker,
               s=20,
               facecolors='none',
               #  edgecolors='k',
               #  edgecolors=np.random.sample(3),
               edgecolors=edgecolors,
               linewidth=0.75,
               zorder=2,
               )
    
        

def gaussian_cdf(x, mu=0, sigma=1):
    return 0.5 * ( 1 + erf( (x - mu) / (sigma*np.sqrt(2)) ) )

def gaussian_pdf(x, mu=0, sigma=1):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
    


# Structural metric wrappers
def ovito_analysis(input_file, args="--centrosymmetry --ackland-jones --acna"):
    """
    Returns (npart, 3) array, where first column is centrosymmetry,
    second column is ackland-jones, third column is adaptive common neighbor analysis.
    For the second and third columns, the legend is 
    0 == OTHER
    1 == FCC
    2 == HCP
    3 == BCC
    4 == ICO
    """
    cmd = "ovito_analysis.py --input_file=%s %s" % (input_file, args)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    data_string = StringIO(result.stdout.decode())
    data = np.loadtxt(data_string)
    return data


def bin_omega(omega, nbins=40, min_hits=5, npart=None, ndim=None):
    num_modes = omega.shape[1]
    n_solids = omega.shape[0]
    n_samples = num_modes * n_solids

    counts, centers, total_hits, _, smallest_bin_width = histogram_log_bins(omega.flatten(), num_of_bins=nbins, min_hits=min_hits)
    # fraction = num_modes / (npart*ndim)
    # counts *= fraction / (n_samples * smallest_bin_width)
    counts /= (smallest_bin_width * n_solids * npart * ndim) 

    return centers, counts


def read_modes(data_folder):
    kappa = []
    e = []

    filenames = ["%s/%s" % (data_folder, fname) for fname in os.listdir(data_folder)]

    for fname in filenames:
        data = np.loadtxt(fname)
        kappa.append(data[:, 0])
        e.append(data[:, 1])

    return np.asarray(kappa), np.asarray(e)
