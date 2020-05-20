import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


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






# def hist_2d_loglog(x, y, num_bins):
#     """bins sets the spatial resolution, should be around 200-500-1000"""
        
#     xmax, xmin, ymax, ymin = np.max(x), np.min(x), np.max(y), np.min(y)
#     logfx, logfy = np.log(xmax/xmin)/num_bins, np.log(ymax/ymin)/num_bins

#     counters = np.zeros((num_bins+1, num_bins+1))
#     for i in range(x.size):
#         x_index = np.floor( np.log(x[i]/xmin) / logfx )
#         y_index = np.floor( np.log(y[i]/ymin) / logfy )
#         counters[x_index, y_index] += 1

#     max_count = np.max(counters[:])
#     print(max_count)

#     counters2 = np.zeros((num_bins+1, num_bins+1))
#     colors = zeros_like(x)
#     select = zeros_like(x)

#     for i in range(x.size):
#         x_index = np.floor( np.log(x[i]/xmin) / logfx )
#         y_index = np.floor( np.log(y[i]/ymin) / logfy )
#         counters2[x_index, y_index] += 1
#         colors[i] = color_map(counters2[x_index, y_index])
#         if counters2[x_index, y_index] > counters[x_index, y_index] - 1:
#             select[i] = 1

        
#         select = np.nonzero(select == 1)
#         x = x[select]
#         y = y[select]
#         colors = colors[select]


# function map = getCloudColorMap(n)
#     map = zeros(n,3);
#     for i=1:n
#         x = (i-1)/n;
#         r = x;
#         g = 0.2*x;
#         b = 0.2*x;
#         map(i,:) = [r g b];
#     end

# function plotColorClouds(x,y,bins)



#     xMax = max(x);
#     xMin = min(x);
#     yMax = max(y);
#     yMin = min(y);
    
#     logfx = log(xMax/xMin)/bins;
#     logfy = log(yMax/yMin)/bins;
    
#     counters = zeros(bins+1,bins+1);
#     maximal = 0;
#     for i=1:length(x)
#         xIndex = floor( log(x(i)/xMin)/logfx )+1;
#         yIndex = floor( log(y(i)/yMin)/logfy )+1;
#         counters(xIndex,yIndex) = counters(xIndex,yIndex)+1;
#         if ( counters(xIndex,yIndex) > maximal )
#             maximal = counters(xIndex,yIndex);
#         end
#     end
    
#     map = getCloudColorMap(maximal);
    
#     counters2 = zeros(bins+1,bins+1);
#     colors = zeros(length(x),3);
#     select = zeros(length(x),1);
    
#     for i=1:length(x)
#         xIndex = floor( log(x(i)/xMin)/logfx )+1;
#         yIndex = floor( log(y(i)/yMin)/logfy )+1;
#         counters2(xIndex,yIndex) = counters2(xIndex,yIndex)+1;
#         colors(i,:) = map(counters2(xIndex,yIndex),:);
#         if ( counters2(xIndex,yIndex) > counters(xIndex,yIndex) - 1 )
#             select(i) = 1;
#         end 
#     end
#     length(x)
#     x = x(select==1);
#     y = y(select==1);
#     %colors = colors(select==1,:)
#     colors = colors(select==1);
    
#     length(x)
    
#     [b,indices] = sort(colors);
#     indices = indices(end:-1:1);
#     x = x(indices);
#     y = y(indices);
#     colors = colors(indices);
    
    
    
#     colormap('Copper');
# %     scatter(x,y,10*ones(length(x),1),sqrt(colors),'filled');
# 	scatter(x,y,15*ones(length(x),1),colors.^(1/3),'filled','s');
# 	%scatter(x,y,12*ones(length(x),1),colors,'filled');
#     set(gca,'xscale','log','yscale','log'); box on;
    
        

def gaussian_cdf(x, mu=0, sigma=1):
    return 0.5 * ( 1 + erf( (x - mu) / (sigma*np.sqrt(2)) ) )

def gaussian_pdf(x, mu=0, sigma=1):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
    





