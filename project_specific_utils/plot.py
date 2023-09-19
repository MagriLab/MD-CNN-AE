'''Some useful plots'''

import numpy as np
import matplotlib.pyplot as plt
import warnings

import typing
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet


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


def hide_contour_lines(*contour_plots: QuadContourSet):
    '''Hide contour lines from contourf plots.'''

    for image in contour_plots:
        for c in image.collections:
            c.set_edgecolor("face")