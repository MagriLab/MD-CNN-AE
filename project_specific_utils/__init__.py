'''
A collection of project specific functions. 
'''
from .plot import create_custom_colormap
my_discrete_cmap = create_custom_colormap(map_name='overleaf-earth',type='discrete')
my_continuous_cmap = create_custom_colormap(map_name='overleaf-earth',type='continuous')
