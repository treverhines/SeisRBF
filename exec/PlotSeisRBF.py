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

def plot_interpolant(D,interp,x,title='',dim=1,ax=None,scatter=False):
  if ax is None:  
    fig,ax = plt.subplots()
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title(title)

  
  buff = 400.0
  N = 150
  minx = np.min(x[:,0])
  maxx = np.max(x[:,0])
  miny = np.min(x[:,1])
  maxy = np.max(x[:,1])
  square = Polygon([(minx-buff,miny-buff),
                    (minx-buff,maxy+buff),
                    (maxx+buff,maxy+buff),
                    (maxx+buff,miny-buff),
                    (minx-buff,miny-buff)])
  ax.add_artist(PolygonPatch(square.difference(D),alpha=1.0,color='k',zorder=1))
  ax.set_xlim((minx-buff,maxx+buff))
  ax.set_ylim((miny-buff,maxy+buff))

  if dim == 1:
    xitp = np.linspace(minx,maxx,N)
    yitp = np.linspace(miny,maxy,N)
    xgrid,ygrid = np.meshgrid(xitp,yitp)
    xflat = xgrid.flatten()
    yflat = ygrid.flatten()
    points = np.zeros((len(xflat),2))
    points[:,0] = xflat
    points[:,1] = yflat
    val = interp(points)
    #val[(np.sqrt(xflat**2+yflat**2) > 6371),:] = 0.0

    im =NonUniformImage(ax,interpolation='bilinear',cmap='cubehelix_r',extent=(minx,maxx,miny,maxy))
    im.set_data(xitp,yitp,np.reshape(val,(N,N)))

    ax.images.append(im)
    if scatter == True:
      p = ax.scatter(x[:,0],
                     x[:,1],
                     c='gray',edgecolor='none',zorder=2,s=10)
    cbar = plt.colorbar(im)

  if dim == 2:
    ax.quiver(x[::3,0],x[::3,1],interp(x)[::3,0],interp(x)[::3,1],color='gray',scale=4000.0,zorder=20)

  return ax

def plot_record_section():
  pass

parser = argparse.ArgumentParser(
           description='''Plots results of SeisRBF''')

parser.add_argument('name',type=str)
parser.add_argument('time',nargs='+',type=float)

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
time = f['time'][...]

D = Polygon(surface_nodes)
N = len(nodes)

S_vel = RBFInterpolant(nodes,eps,value=np.sqrt(np.abs(mu_itp(nodes))/rho_itp(nodes)))
P_vel = RBFInterpolant(nodes,eps,value=np.sqrt((lam_itp(nodes)+2*mu_itp(nodes))/rho_itp(nodes)))
for t in args.time:
  timeidx = np.argmin(np.abs(time - t))
  soln = RBFInterpolant(nodes,eps,alpha=alpha[timeidx,:,:])
  mag = RBFInterpolant(nodes,eps,value=np.sqrt(np.sum(soln(nodes)**2,1)))
  ax = plot_interpolant(D,mag,nodes,title='displacement (meters) at time %s seconds' % time[timeidx],dim=1)
  ax = plot_interpolant(D,soln,nodes,dim=2,ax=ax)
  #ax = plot_interpolant(D,lam_itp,nodes,title='second lame parameter (lambda)',dim=1,scatter=True)
  #ax = plot_interpolant(D,mu_itp,nodes,title='first lame parameter (mu)',dim=1,scatter=True)

ax = plot_interpolant(D,rho_itp,nodes,title='density (kg/m**3)',dim=1,scatter=True)
ax = plot_interpolant(D,S_vel,nodes,title='S wave velocity (m/s)',dim=1,scatter=True)
ax = plot_interpolant(D,P_vel,nodes,title='P wave velocity (m/s)',dim=1,scatter=True)

#fig,ax = plot_displacement_magnitude(D,nodes,eps,alpha[timeidx,:,:],title='time: %s s' % time[timeidx])
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
