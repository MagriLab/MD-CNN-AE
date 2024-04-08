'''Some useful plots'''

import numpy as np
import matplotlib.pyplot as plt
import warnings

import typing
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
import matplotlib.colors as mcolors
from functools import partial

# user-defined color maps
discrete_dict = {
    'defne': ["#03BDAB", "#FEAC16", "#5D00E6","#F2BCF3","#AFEEEE"],
    'overleaf-earth': ['#1e446b','#3c5920','#26aa08','#bf9000','#ffc000'],
}
continuous_dict = {
    'defne': ["#03BDAB", "#FEAC16", "#5D00E6"],
    'overleaf-earth': ['#1e446b','#26aa08','#bf9000','#ffc000'],
}


def scatter_matrix(data:typing.Union[np.ndarray,list], 
                    variable_names:list = [], 
                    space:list = [0, 0], 
                    kwargs_figure:dict = {}, 
                    kwargs_scatter:dict = {}, 
                    kwargs_hist:dict = {}) -> typing.Tuple[Figure, Axes]:
    '''Plot the scatter matrix of data.
    
    Create matrix of plots, with histograms of variables on the diagonal and scatter plots of one variable against another on the rest of the plots. Return the figure and axes objects.

    Arguments:
        data: numpy array_like, each column is a variable and each row is a sample of the variables.
        variable_names: list of variable names, must match the number of columns of data. If not provided, the figure will be unlabled.
        space: list [wspace, hspace] between subplots, default no space.
        kwargs_figure: dictionary of arguments for pyplot.figure.
        kwargs_scatter: arguments for pyplot.scatter.
        kwargs_hist: arguments for pyplot.hist.
    
    Returns:
        fig: pyplot.figure.Figure
        ax: an array of Axes, with dimension [nvar, nvar], where nvar is the number of variables
    '''
    data = np.array(data)
    nvar = data.shape[1] # number of variables
    n = data.shape[0] # number of samples

    if data.ndim != 2:
        raise ValueError('To plot a scatter matrix, the input must be a matrix where each column is a variable.')
    if nvar > n:
        warnings.warn('Number of variables exceed the number of samples, proceed with caution.')
    if (len(variable_names) != 0) and (nvar != len(variable_names)):
        raise ValueError('Number of variable names does not match the number of variables.')
    
    fig, ax = plt.subplots(nvar, nvar, sharex='col', sharey='row', **kwargs_figure)

    sharey = ax[0,0].get_shared_y_axes()
    ax[0,nvar-1].tick_params(right=True, labelright=True)
    
    for i in range(nvar):
        if len(variable_names) > 0:
            ax[i,0].set_ylabel(variable_names[i])
            ax[-1,i].set_xlabel(variable_names[i])
        for j in range(nvar):
            if i == j: # histogram on the diagonal
                sharey.remove(ax[i,j])
                ax[i,j].hist(data[:,i], **kwargs_hist)
                ax[i,j].tick_params(which='both', axis='y', left=False, labelleft=False)
            else:
                ax[i,j].scatter(data[:,j], data[:,i], **kwargs_scatter)
    fig.subplots_adjust(wspace=space[0], hspace=space[1])

    return fig, ax


# https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills
def hide_contour_lines(*contour_plots: QuadContourSet):
    '''Hide contour lines from contourf plots.'''

    for image in contour_plots:
        for c in image.collections:
            c.set_edgecolor("face")


def create_discrete_colormap(colors, name='custom_colormap'):
    """Create a discrete colormap from given color hex codes.
    
    Args:
        colors (list): List of color hex codes.
        name (str, optional): Name of the colormap. Defaults to 'custom_colormap'.
    
    Returns:
        matplotlib.colors.ListedColormap: The discrete colormap object.
    """
    cmap = mcolors.ListedColormap(colors, name=name)
    return cmap

def create_continuous_colormap(colors, name='custom_colormap', N=256):
    """Create a continuous colormap from given color hex codes.
    
    Args:
        colors (list): List of color hex codes.
        name (str, optional): Name of the colormap. Defaults to 'custom_colormap'.
        N (int, optional): Number of color levels. Defaults to 256.
    
    Returns:
        matplotlib.colors.ListedColormap: The continuous colormap object.
    """
    ncolors = len(colors)
    if ncolors < 2:
        raise ValueError("Please provide at least two colors.")

    color_array = np.zeros((N, 4))
    for i in range(N):
        idx1 = int(i * (ncolors - 1) / N)
        idx2 = min(idx1 + 1, ncolors - 1)
        t = i * (ncolors - 1) / N - idx1
        color_array[i] = tuple((1 - t) * c1 + t * c2 for c1, c2 in zip(mcolors.to_rgba(colors[idx1]), mcolors.to_rgba(colors[idx2])))
    cmap = mcolors.ListedColormap(color_array, name=name)
    return cmap

def create_custom_colormap(map_name='defne',type='discrete', colors=None, N=256):
    """Create a custom colormap.

    This function creates either a discrete or continuous colormap based on the given parameters.

    Args:
        map_name (str, optional): Name of the custom colormap. Defaults to 'defne'.
        cmap_type (str, optional): Type of the colormap ('discrete' or 'continuous'). Defaults to 'discrete'.
        colors (list, optional): List of color hex codes. If None, uses predefined colormap based on map_name. Defaults to None.
        N (int, optional): Number of color levels for continuous colormap. Defaults to 256.

    Returns:
        matplotlib.colors.ListedColormap: The custom colormap object.
    """
    colors_dict = {'discrete': discrete_dict, 'continuous': continuous_dict}
    function_dict = {'discrete': create_discrete_colormap, 'continuous': partial(create_continuous_colormap, N=N)}
    if colors:
        assert isinstance(colors, list)
        colors_hex = colors
    else:
        try:
            colors_hex = colors_dict[type][map_name]
        except KeyError:
            print(f'map {map_name} does not exist.')
            raise NotImplementedError
    cmap_fn = function_dict[type]
    cmap = cmap_fn(colors_hex, map_name)
    return cmap