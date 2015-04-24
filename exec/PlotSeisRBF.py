#!/usr/bin/env python
import sys
import modules.project_plot as project_plot
import pickle
import sys
import SeisRBF
import matplotlib.pyplot as plt
import h5py
from modules.halton import Halton
from modules.radial import RBFInterpolant
from shapely.geometry import Polygon
from descartes import PolygonPatch 
from matplotlib.image import NonUniformImage 
import argparse
import numpy as np

def plot_displacement_magnitude(D,nodes,eps,alpha,title=''):
  buff = 200  
  N = 100

  soln = RBFInterpolant(nodes,eps,alpha=alpha)
  fig,ax = plt.subplots()
  ax.set_aspect('equal',adjustable='box')
  p = ax.scatter(nodes[:,0],nodes[:,1],
                 c='gray',edgecolor='none',
                 zorder=2,s=10)
  ax.set_title(title)

  minx = np.min(nodes[:,0])
  maxx = np.max(nodes[:,0])
  miny = np.min(nodes[:,1])
  maxy = np.max(nodes[:,1])
  xitp = np.linspace(minx,maxx,N)
  yitp = np.linspace(miny,maxy,N)
  xgrid,ygrid = np.meshgrid(xitp,yitp)
  xflat = xgrid.flatten()
  yflat = ygrid.flatten()
  val = soln(zip(xflat,yflat))
  val = np.sqrt(np.sum(val**2,1))

  ax.set_xlim((minx-buff/2,maxx+buff/2))
  ax.set_ylim((miny-buff/2,maxy+buff/2))
  square = Polygon([(minx-buff,miny-buff),
                    (minx-buff,maxy+buff),
                    (maxx+buff,maxy+buff),
                    (maxx+buff,miny-buff),
                    (minx-buff,miny-buff)])
  
  ax.add_artist(PolygonPatch(square.difference(D),alpha=1.0,color='k',zorder=1))
  im =NonUniformImage(ax,interpolation='bilinear',
                         cmap='cubehelix_r',
                         extent=(minx,maxx,miny,maxy))
  im.set_data(xitp,yitp,np.reshape(val,(N,N)))
  ax.images.append(im)
  return fig,ax


parser = argparse.ArgumentParser(
           description='''Plots results of SeisRBF''')

parser.add_argument('name',type=str)
parser.add_argument('time',type=float)

args = parser.parse_args()
H = Halton(3)
interp_nodes = 1000
name = args.name
f = h5py.File('output/%s/%s.h5' % (args.name,args.name),'r')

nodes = f['nodes'][...]
eps = f['epsilon'][...]
surface_nodes = nodes[f['surface_index'][...]]
interior_nodes = nodes[f['interior_index'][...]]
alpha = f['alpha'][...]
mu_itp = RBFInterpolant(nodes,eps,value=f['mu'][...])
rho_itp = RBFInterpolant(nodes,eps,value=f['rho'][...])
lam_itp = RBFInterpolant(nodes,eps,value=f['lambda'][...])
time = f['time']
D = Polygon(surface_nodes)
N = len(nodes)


fig,ax = plot_displacement_magnitude(D,nodes,eps,alpha[10,:,:],title='')
'''
xinterp = SeisRBF.pick_nodes(H,D,lambda x:1,interp_nodes)

anim1 = project_plot.animate_displacement(D,
                                          xinterp,
                                          zip(soln_itp[::10],time[::10]))

anim2 = project_plot.animate_displacement_magnitude(D,
                                                    nodes,
                                                    soln_itp[::10],
                                                    time[::10])

fig = project_plot.plot_seismogram(zip(soln_itp,time),
                                   nodes[10])

fig = project_plot.plot_record_section(zip(soln_itp,time),
                                       surface_nodes[:N//2])

fig = project_plot.plot_interpolant(D,mu_itp,nodes)
'''
plt.show()
