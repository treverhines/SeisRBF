#!/usr/bin/env python
import mayavi.mlab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from descartes import PolygonPatch
from matplotlib.image import NonUniformImage
from shapely.geometry import Polygon

# Plot displacements
def _update1(itr,x,soln,p,t):
  u = soln[itr][0](x)
  if itr == 0:
    p.set_UVC(0.0*u[:,0],0.0*u[:,1])
    t.set_text('')
  else:
    p.set_UVC(u[:,0],u[:,1])
    t.set_text('time: %04d s' % soln[itr][1])

  return p,t


def _update3(itr,D,x,soln,t,ax,time):
  N = 100
  #ax.clear()
  buff = 200.0
  minx = np.min(x[:,0])
  maxx = np.max(x[:,0])
  miny = np.min(x[:,1])
  maxy = np.max(x[:,1])
  ax.set_xlim((minx-buff/2,maxx+buff/2))
  ax.set_ylim((miny-buff/2,maxy+buff/2))
  square = Polygon([(minx-buff,miny-buff),
                    (minx-buff,maxy+buff),
                    (maxx+buff,maxy+buff),
                    (maxx+buff,miny-buff),
                    (minx-buff,miny-buff)])
  ax.add_artist(PolygonPatch(square.difference(D),alpha=1.0,color='k',zorder=1))
  xitp = np.linspace(minx,maxx,N)
  yitp = np.linspace(miny,maxy,N)
  xgrid,ygrid = np.meshgrid(xitp,yitp)
  xflat = xgrid.flatten()
  yflat = ygrid.flatten()
  ax.images = []
  im =NonUniformImage(ax,interpolation='bilinear',
                         cmap='cubehelix',
                         extent=(minx,maxx,miny,maxy))
  val = soln[itr](zip(xflat,yflat))
  val = np.sqrt(np.sum(val**2,1))
  im.set_data(xitp,yitp,np.reshape(val,(N,N)))  
  ax.images.append(im)
  t.set_text('t: %s s' % time[itr])
  return ax,t

def animate_displacement_magnitude(D,x,soln,time):
  fig,ax = plt.subplots()
  plt.gca().set_aspect('equal', adjustable='box')
  #ax.add_artist(PolygonPatch(D,alpha=0.2,color='k',zorder=0))
  p = ax.scatter(x[:,0],
                 x[:,1],
                 c='gray',edgecolor='none',zorder=2,s=10)

  t = ax.text(4000.0,5000.0,'',fontsize=12,color='white')
  anim = animation.FuncAnimation(fig,
                                 _update3,
                                 fargs=(D,x,soln,t,ax,time),
                                 frames=len(soln),
                                 interval=20,
                                 blit=True)
  return anim

def animate_displacement(D,x,soln):
  fig,ax = plt.subplots()
  plt.gca().set_aspect('equal', adjustable='box')
  ax.add_artist(PolygonPatch(D,alpha=0.2,color='k',zorder=0))
  u_o = soln[0][0](x)
  p = ax.quiver(x[:,0],x[:,1],u_o[:,0],u_o[:,1],scale=100.0)
  t = ax.text(4000.0,6000.0,'',fontsize=16)
  anim = animation.FuncAnimation(fig,
                                 _update1,
                                 fargs=(x,soln,p,t),
                                 frames=len(soln),
                                 interval=20,
                                 blit=True)
  return anim


# Plot difference between acceleration and F(u)
def _update2(itr,x,soln,A,p,t):
  if (itr == 0) | (itr >= (len(soln)-1)):
    u = soln[itr][0](x)
    p.set_UVC(0.0*u[:,0],0.0*u[:,1])
    t.set_text('')

  else:
    alpha = soln[itr][0].alpha
    a = np.einsum('ijkl,kl',A,alpha)
    dt = soln[itr][1] - soln[itr-1][1]
    a_euler = (soln[itr-1][0](x) - 2*soln[itr][0](x) + soln[itr+1][0](x))/dt**2
    error = a - a_euler
    p.set_UVC(error[:,0],error[:,1])
    t.set_text('time: %04d s' % soln[itr][1])

  return p,t

def animate_error(D,x,soln,A):
  fig,ax = plt.subplots()
  plt.gca().set_aspect('equal', adjustable='box')
  ax.add_artist(PolygonPatch(D,alpha=0.2,color='k',zorder=0))
  u_o = soln[0][0](x)
  p = ax.quiver(x[:,0],x[:,1],0*u_o[:,0],0*u_o[:,1],scale=0.001)
  t = ax.text(4000.0,6000.0,'',fontsize=16)
  anim = animation.FuncAnimation(fig,
                                 _update2,
                                 fargs=(x,soln,A,p,t),
                                 frames=len(soln),
                                 interval=20,
                                 blit=True)
  return anim

def plot_interpolant(D,interp,x,title='figure'):
  buff = 100.0
  fig,ax = plt.subplots()
  plt.gca().set_aspect('equal', adjustable='box')

  plt.title(title,fontsize=16)

  N = 200
  minx = np.min(x[:,0])
  maxx = np.max(x[:,0])
  miny = np.min(x[:,1])
  maxy = np.max(x[:,1])
  xitp = np.linspace(minx,maxx,N)
  yitp = np.linspace(miny,maxy,N)
  xgrid,ygrid = np.meshgrid(xitp,yitp)
  xflat = xgrid.flatten()
  yflat = ygrid.flatten()
  points = np.zeros((len(xflat),2))
  points[:,0] = xflat
  points[:,1] = yflat
  val = interp(points)
  val[(np.sqrt(xflat**2+yflat**2) > 6371),:] = 0.0

  square = Polygon([(minx-buff,miny-buff),
                    (minx-buff,maxy+buff),
                    (maxx+buff,maxy+buff),
                    (maxx+buff,miny-buff),
                    (minx-buff,miny-buff)])

  #help(D)
  im =NonUniformImage(ax,interpolation='bilinear',cmap='cubehelix',extent=(minx,maxx,miny,maxy))
  im.set_data(xitp,yitp,np.reshape(val,(N,N)))
  
  ax.images.append(im)
  ax.add_artist(PolygonPatch(square.difference(D),alpha=1.0,color='k',zorder=1))
  p = ax.scatter(x[:,0],
                 x[:,1],
                 c='gray',edgecolor='none',zorder=2,s=10)
  cbar = plt.colorbar(im)
  cbar.ax.set_ylabel(title)
  ax.set_xlim((minx-buff,maxx+buff))
  ax.set_ylim((miny-buff,maxy+buff))
  #fig.colorbar(p)
  return fig

def plot_seismogram(soln,x):
  fig = plt.figure()
  ax1 = fig.add_subplot(211)
  ax1.set_title('station x:%g, y:%g' % (x[0],x[1]))  
  ax2 = fig.add_subplot(212)
  u_list = []
  v_list = []
  t_list = []
  for s,t in soln:
    u,v = s([x])[0]
    t_list += [t] 
    u_list += [u] 
    v_list += [v] 

  ax1.plot(t_list,u_list)
  ax2.plot(t_list,v_list)
  
  return fig

def plot_record_section(soln,x_list):
  fig,ax = plt.subplots()
  for x in x_list:
    theta = np.arctan2(x[0],x[1])*(180/np.pi)
    u_list = []
    v_list = []
    t_list = []
    for s,t in soln:
      u,v = s([x])[0]
      t_list += [t] 
      u_list += [u] 
      v_list += [v] 

    t_list = np.array(t_list)
    u_list = np.array(u_list)
    v_list = np.array(v_list)
    ax.plot(theta+0.1*u_list,t_list/60.0,'k')
  
  return fig





