from curses.panel import bottom_panel
from curses.textpad import rectangle
from fractions import Fraction
from matplotlib import gridspec, pyplot as plt
import mpl_toolkits.axes_grid1 as grid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import Divider
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
import numpy as np
from pyparsing import alphas
import matplotlib as mpl

mpl.rcParams['ytick.labelsize'] = 8


img = np.reshape(np.arange(30),(5,6))

fig = plt.figure(constrained_layout = False,figsize=(7,4.5))
subfig = fig.subfigures(2,1,wspace=0.1,height_ratios=[1,1])
subfig[0].set_facecolor('0.65')
subfig[0].suptitle('0')

gs0 = subfig[0].add_gridspec(nrows=3,ncols=1,left=0.02,right=0.98,top=1,bottom=0,wspace=0.05,height_ratios=[0.22,1,1],)
subfig0 = []
subfig0.append(subfig[0].add_subfigure(gs0[1]))
subfig0.append(subfig[0].add_subfigure(gs0[2]))
subfig0[0].suptitle('v',x=0.02,y=0.5)
subfig0[0].set_facecolor('0.75')

gs00=subfig0[0].add_gridspec(nrows=1,ncols=6,left=0.02,right=0.98,top=0.8,width_ratios=[0.2,1,0.02,1,0.02,3])
subfig00=[]
subfig00.append(subfig0[0].add_subfigure(gs00[1]))
subfig00.append(subfig0[0].add_subfigure(gs00[3]))
subfig00.append(subfig0[0].add_subfigure(gs00[5]))

subfig00[0].set_facecolor('0.85')
subfig00[0].suptitle('Inst.',fontsize='medium')
ax000 = subfig00[0].subplots(1,1,subplot_kw={'xticks':[],'yticks':[],'aspect':0.50})
im000 = ax000.imshow(img)
plt.colorbar(im000,ax=ax000,shrink=0.9)

subfig00[1].set_facecolor('0.85')
subfig00[1].suptitle('Mean',fontsize='medium')
ax001 = subfig00[1].subplots(1,1,subplot_kw={'xticks':[],'yticks':[]})
im001 = ax001.imshow(img)
plt.colorbar(im001,ax=ax001,shrink=0.9)

subfig00[2].set_facecolor('0.85')



subfig[1].set_facecolor('0.65')
subfig[1].suptitle('1')


plt.show()

# left=0.01, right=0.99,
# 
# 