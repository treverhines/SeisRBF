#!/usr/bin/env python
#
# AOSS 555
# Final Project
# Trever Hines
#
# Description
# -----------
# This program solves the equation of motion for two-dimensional
# displacement in a two dimensional elastic domain.  

# I assume a disk shaped domain, although this can be readily changed.
# This problem is intended to simulate seismic waves on a global scale
# and I assume elastic properties and densities from the Preliminary
# Reference Earth Model (PREM) (Dziewonski & Anderson 1981). 

# I use radial basis functions to solve this problem because the
# heterogeneous material properties require a heterogeneous node
# density

# Dependencies 
# ------------ 
# In additoon to the standard scientific python packages, this program
# uses three other modules which I wrote: 
#
#   radial.py: Contains the radial basis functions 
#   halton.py: Used to create a low discrepancy node distribution
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from modules.radial import mq 
from modules.radial import RBFInterpolant 
from modules.halton import Halton
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
from descartes import PolygonPatch
from modules.misc import Timer
from modules.misc import timestamp
import pickle
import logging

logger = logging.getLogger()
#logging.basicConfig(level=logging.DEBUG)
T = Timer()

# Run Parameters
# ----------------- 
NODES = 4000 # number of total nodes 
BOUNDARY_NODES = 200 # number of boundary nodes
INTERP_NODES = 1000 # number of interpolation points for plotting 
RBF = mq # multiquadtratic radial basis function
SOLVER = np.linalg.solve # this is pythons default solver
DIM = 2 # number of spatial dimensions
TMAX = 10.0 # max time
DT = 2.0 # time step
SAVE_INTERVAL = 10 # time steps between saves
EPS = 1.0 # shape parameter scale 
R_EARTH = 6371.0 #km
EPICENTER = np.array([0.0,5000.0])
COUPLE_DISTANCE = 500.0 # km
# moment tensor has units of N m
MOMENT_TENSOR = np.array([[1.e21,0.0],
                          [0.0,1.e21]])

# User functions
# --------------
def initial_conditions(x):
  '''
  Initial displacement an velocity field.
  '''
  N = len(x)
  u = np.zeros((N,DIM))
  v = np.zeros((N,DIM))
  return u,v

def source_time_function(t):
  '''
  a function describing the time history of the force term
  
  This function should integrate to 1 on the interval 0 to inf
  '''
  if t < 20.0:
    return 0.1

  else:
    return 0.0 

def boundary(t):
  '''
  parameterized boundary function 
  
  Parameters
  ----------
    t: array with values between 0 and 1

  Returns
  -------
    coordinates of the boundary 

  '''
  p = np.linspace(0,2*np.pi,1000)
  x = R_EARTH*np.sin(p)
  y = R_EARTH*np.cos(p)
  curve = LineString(zip(x,y))
  
  if hasattr(t,'__iter__'):
    out = np.zeros((len(t),2))
    for itr,val in enumerate(t):
      out[itr] = np.array(curve.interpolate(val,normalized=True))

  else:
    out = np.array(curve.interpolate(t,normalized=True))

  return out


def node_density(x):
  '''
  returns a value between 0 and 1 indicating the desired density
  of nodes at the given points

  Parameters
  ----------
    x: N by 2 array of coordinates

  Returns
  -------
    length N array of node densities between 0 and 1    

  '''
  layer_depths = [[np.inf,6350],
                  [6350,6000],
                  [6000,5600],
                  [5600,3800],
                  [3800,3200],
                  [3200,1500],
                  [1500,1000],
                  [1000,0]]

  layer_density = [0.4,
                   0.4,
                   0.4,
                   0.4,
                   1.0,
                   0.4,
                   0.4,
                   0.4]
  
  R = np.sqrt(np.sum(x**2,1))
  out = np.zeros(len(R))
  for depths,density in zip(layer_depths,layer_density):
    out[(R<=depths[0])&(R>depths[1])] = density  

  return out

# Seismic Source Functions
# ------------------------
def delta(x,x_o,eps):
  '''
  returns an 2 dimensional Gaussian function with standard deviation
  equal to eps
  '''
  exponent = np.sum(-((x-x_o)/(2*eps))**2,1)
  return np.exp(exponent)/(2*np.pi*eps**2) 


def force_couple(x,x_o,i,j,d):
  '''
  returns a spatial distribution of forces resulting from component
  (i,j) of the moment tensor.  
  
  Parameters
  ----------
    x: output locations
    x_o: center of the force couple
    i,j: force couple components
    d: distance between the couples

  Returns
  -------
    forces at positions x
  '''
  force_out = np.zeros((len(x),DIM))
  x_1 = 1.0*x_o 
  x_1[j] += d/2.0
  x_2 = 1.0*x_o 
  x_2[j] -= d/2.0
  force_out[:,i] +=  delta(x,x_1,d/4.0)
  force_out[:,i] += -delta(x,x_2,d/4.0)
  return force_out
    

def source(x,t):
  '''
  This function uses the global MOMENT_TENSOR variable and the 
  user defined source_time_function to produce the force at position
  x and time t
  '''
  # I want a source that is equivalent to a Mw8 earthquake.  This is
  # 10^21 Nm moment
  d = COUPLE_DISTANCE
  out = MOMENT_TENSOR[0,0]*force_couple(x,EPICENTER,0,0,d)*1e-6
  out += MOMENT_TENSOR[1,1]*force_couple(x,EPICENTER,1,1,d)*1e-6
  out += MOMENT_TENSOR[0,1]*force_couple(x,EPICENTER,0,1,d)*1e-6
  out += MOMENT_TENSOR[1,0]*force_couple(x,EPICENTER,1,0,d)*1e-6
  return out*source_time_function(t)


# Numerical Functions
# -------------------
def leapfrog(F,u,v,dt,a=None,F_args=None,F_kwargs=None):
  '''
  Leapfrog time stepping algorithm which solves the next
  time step in d^u/dt^2 = F(u)
  
  Parameters 
  ---------- 
    F: Funtion which returns both the acceleration at the current time
      step and the spectral coefficients of u. This function takes u
      as the first argument and then F_args and F_kwargs
    u: displacement at the current time step 
    v: velocity at the current time step 
    dt: time step size
    a: (optional) acceleration at the current time step. If this is
      not provided then a will be computed with F(u). However, this
      is inefficient because F(u) should have been computed at the 
      previous time step
    F_args: (optional) tuple of dditional arguments to F
    F_kwargs: (optional) dictionary of additional key word arguments 
      to F

  Returns
  -------
    tuple of u_new,v_new,a_new, and alpha_new.  These are the 
    displacements, velocities, accelerations, and spectral 
    coefficients for the new time step
  '''
  if F_args is None:
    F_args = ()
  if F_kwargs is None:
    F_kwargs = {}

  if a is None:
    a,alpha = F(u,*f_args,**f_kwargs)

  u_new = u + v*dt + 0.5*a*dt**2
  a_new,alpha_new = F(u_new,*F_args,**F_kwargs)
  v_new = v + 0.5*(a + a_new)*dt
  return u_new,v_new,a_new,alpha_new


def acceleration_matrix(x,c,eps,lam_itp,mu_itp,rho_itp):
  '''
  forms a matrix which returns accelerations at points x when
  multiplied by the spectral coefficients

  Paramters
  ---------
  x_ij: shape (N,DIM) array of points
  c_ij: shape (M,DIM) array of RBF centers
  eps_i: shape (M,) array of shape parameters
  lam_itp: RBFInterpolant for lambda
  mu_itp: RBFInterpolant for mu
  rho_itp: RBFInterpolant for rho

  Returns
  -------
  A_ijkl: shape (N,DIM,M,DIM) array. Organized such that 
    a_ij = A_ijkl*alpha_kl, where a_ij is the acceleration at point i 
    in direction j, while alpha_kl is the spectral coefficient k for 
    displacement in direction l

  '''
  N = len(x)
  M = len(c)
  A = np.zeros((N,DIM,M,DIM))
  A[:,0,:,0] = ((lam_itp(x,diff=(1,0)) + 2*mu_itp(x,diff=(1,0)))*RBF(x,c,eps,diff=(1,0)) + 
                (lam_itp(x,diff=(0,0)) + 2*mu_itp(x,diff=(0,0)))*RBF(x,c,eps,diff=(2,0)) +
                mu_itp(x,diff=(0,1))*RBF(x,c,eps,diff=(0,1)) +
                mu_itp(x,diff=(0,0))*RBF(x,c,eps,diff=(0,2)))/rho_itp(x)

  A[:,0,:,1] = (lam_itp(x,diff=(1,0))*RBF(x,c,eps,diff=(0,1)) + 
                lam_itp(x,diff=(0,0))*RBF(x,c,eps,diff=(1,1)) + 
                mu_itp(x,diff=(0,1))*RBF(x,c,eps,diff=(1,0)) + 
                mu_itp(x,diff=(0,0))*RBF(x,c,eps,diff=(1,1)))/rho_itp(x)

  A[:,1,:,0] = (mu_itp(x,diff=(1,0))*RBF(x,c,eps,diff=(0,1)) + 
                mu_itp(x,diff=(0,0))*RBF(x,c,eps,diff=(1,1)) + 
                lam_itp(x,diff=(0,1))*RBF(x,c,eps,diff=(1,0)) + 
                lam_itp(x,diff=(0,0))*RBF(x,c,eps,diff=(1,1)))/rho_itp(x)

  A[:,1,:,1] = ((lam_itp(x,diff=(0,1)) + 2*mu_itp(x,diff=(0,1)))*RBF(x,c,eps,diff=(0,1)) + 
                (lam_itp(x,diff=(0,0)) + 2*mu_itp(x,diff=(0,0)))*RBF(x,c,eps,diff=(0,2)) +
                mu_itp(x,diff=(1,0))*RBF(x,c,eps,diff=(1,0)) +
                mu_itp(x,diff=(0,0))*RBF(x,c,eps,diff=(2,0)))/rho_itp(x)

  return A


def traction_matrix(x,n,c,eps,lam_itp,mu_itp):
  '''
  forms a matrix which returns traction force at points x and normal n 
  when multiplied by the spectral coefficients  

  Paramters
  ---------
  x_ij: shape (N,DIM) array of locations
  n_ij: shape (N,DIM) array of normal directions 
  c_ij: shape (M,DIM) array of RBF centers 
  eps_i: shape (M,) array of shape parameters
  lam_itp: RBFInterpolant for lambda
  mu_itp: RBFInterpolant for mu

  Returns
  -------
  T_ijkl: shape (N,DIM,M,DIM) array. Organized such that 
    tau_ij = T_ijkl*alpha_kl, where tau_ij is the traction force in 
    direction j at point i with normal direction i, while alpha_kl is 
    the spectral coefficient k for displacement in direction l

  '''
  N = len(x)
  M = len(c)
  T = np.zeros((N,DIM,M,DIM))
  T[:,0,:,0] = (n[:,[0]]*(lam_itp(x) + 2*mu_itp(x))*RBF(x,c,eps,diff=(1,0)) +
                n[:,[1]]*mu_itp(x)*RBF(x,c,eps,diff=(0,1)))

  T[:,0,:,1] = (n[:,[0]]*lam_itp(x)*RBF(x,c,eps,diff=(0,1)) + 
                n[:,[1]]*mu_itp(x)*RBF(x,c,eps,diff=(1,0)))

  T[:,1,:,0] = (n[:,[0]]*mu_itp(x)*RBF(x,c,eps,diff=(0,1)) + 
                n[:,[1]]*lam_itp(x)*RBF(x,c,eps,diff=(1,0)))

  T[:,1,:,1] = (n[:,[1]]*(lam_itp(x) + 2*mu_itp(x))*RBF(x,c,eps,diff=(0,1)) +
                n[:,[0]]*mu_itp(x)*RBF(x,c,eps,diff=(1,0)))

  return T


def interpolation_matrix(x,c,eps):
  '''
  forms a matrix which returns displacement at points x when 
  multiplied by the spectral coefficients  

  Paramters
  ---------
  x_ij: shape (N,DIM) array of locations
  c_ij: shape (M,DIM) array of RBF centers 
  eps_i: shape (M,) array of shape parameters

  Returns
  -------
  G_ijkl: shape (N,DIM,M,DIM) array. Organized such that 
    u_ij = G_ijkl*alpha_kl, where u_ij is the displacement in 
    direction j at point i, while alpha_kl is the spectral coefficient 
    k for displacement in direction l

  '''
  N = len(x)
  M = len(c)
  G = np.zeros((N,DIM,M,DIM))
  G[:,0,:,0] = RBF(x,c,eps)
  G[:,1,:,1] = RBF(x,c,eps)

  return G


def nearest(x):
  '''                
  returns a list of distances to the nearest point for each point in     
  x. This is used to determine the shape parameter for each radial      
  basis function     
  '''
  tol = 1e-4
  x = np.asarray(x)
  if len(np.shape(x)) == 1:
    x = x[:,None]

  N = len(x)
  A = (x[None] - x[:,None])**2
  A = np.sqrt(np.sum(A,2))
  A[range(N),range(N)] = np.max(A)
  nearest_dist = np.min(A,1)
  nearest_idx = np.argmin(A,1)
  if any(nearest_dist < tol):
    logger.WARNING('at least one node is a duplicate or very close to '
                   'another node')

  return nearest_dist,nearest_idx


def pick_nodes(H,D,R,N):
  '''
  picks N nodes based on the provided halton sequence, H, the domain, D, 
  and the node density function, R.
  '''
  count = 0
  out = np.zeros((0,2))
  while count < N:
    x = H.qunif(-R_EARTH,R_EARTH,4*N)[:,:2]
    nd = R(x)
    accept_bool = nd > H(4*N)[:,2]
    accept = np.nonzero(accept_bool)[0]
    x = x[accept]
    accept = []
    for idx,val in enumerate(x):
      p = Point(val)
      if D.contains(p):
        accept += [idx]

    accept = np.array(accept)
    accept = accept[:(N-count)]
    x = x[accept]
    out = np.vstack((out,x))
    count += len(x)

  return out


def F(u,G,A,f,BCidx,BCval): 
  '''
  computes acceleration from u.  This is done by first finding the 
  spectral coefficients from u and then using the spectral 
  coefficients to evaluate the derivative. The matrix G is the 
  interpolation matrix except that rows BCidx have been swapped out
  with rows that will enforce the boundary conditions.  The 
  corresponding rows in u get replaced with BCval in this function

  '''
  # set boundary conditions
  u[BCidx,:] = BCval  
  
  # Reshape the arrays into something tractable
  N,D1,M,D2 = np.shape(G)
  u_shape = (N,D1)
  assert np.shape(u) == u_shape
  alpha_shape = (M,D2)
  G = np.reshape(G,(N,D1,M*D2))
  # ijm
  G = np.einsum('ijm->mij',G)
  G = np.reshape(G,(M*D2,N*D1))
  # mn
  G = np.einsum('mn->nm',G)
  u = np.reshape(u,N*D1)
  T.toc()

  T.tic('solve for alpha')
  # solve for alpha
  alpha = SOLVER(G,u)
  T.toc()

  # reshape alpha into a 2D array
  alpha = np.reshape(alpha,alpha_shape)

  # solve for acceleration
  a = np.einsum('ijkl,kl',A,alpha)

  # add acceleration due to forcing term
  a = a + f

  return a,alpha


def main():
  # setup logger
  name = timestamp()
  os.makedirs('output/%s' % name)
  formatter = logging.Formatter('%(asctime)s %(module)s: [%(levelname)s] %(message)s',
                                '%m/%d/%Y %H:%M:%S')
  file_handler = logging.FileHandler('output/%s/%s.log' % (name,name),'w')
  stream_handler = logging.StreamHandler(sys.stdout)
  file_handler.setFormatter(formatter)
  stream_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  logger.addHandler(stream_handler)
  logger.setLevel(logging.INFO)
  logger.info('run name: %s' % name)

  T = Timer()
  T.tic('Define Topology')
  # Define Topology
  # ---------------
  # Keep track of the total number of nodes
  node_count = 0
  total_nodes = np.zeros((0,2))

  # initiate a Halton sequence 
  H = Halton(dim=3)

  # define boundary nodes
  # NOTE: this should become a user defined parameter
  t = np.linspace(0,1,BOUNDARY_NODES+1)[:-1]
  surface_nodes = boundary(t)
  surface_idx = range(node_count,len(surface_nodes)+node_count)
  total_nodes = np.vstack((total_nodes,surface_nodes))
  node_count += len(surface_nodes)  

  # find the normal vectors for each surface node, this is used for 
  # applying the boundary conditions
  surface_tangents = (surface_nodes[range(1,BOUNDARY_NODES)+[0],:] - 
                      surface_nodes[range(BOUNDARY_NODES),:])
  surface_normals = np.zeros((BOUNDARY_NODES,DIM))
  surface_normals[:,0] = surface_tangents[:,1]
  surface_normals[:,1] = -surface_tangents[:,0]
  normal_lengths = np.sqrt(np.sum(surface_normals**2,1))
  surface_normals = surface_normals/normal_lengths[:,None]
  
  # Define polygon based on boundary nodes
  D = Polygon(surface_nodes)

  # Add a buffer so that the polygon is slightly smaller than what is 
  # defined by the boundary nodes. This is needed for a stable 
  # solution.  The buffer is the minimum of the distances between 
  # adjacent surface nodes
  buffer_size = np.max(nearest(surface_nodes)[0])
  D = D.buffer(-buffer_size)
  assert D.is_valid

  # Find interior nodes that are within the domain and adhere to the
  # user specified node density
  interior_nodes = pick_nodes(H,D,node_density,NODES-node_count)
  interior_idx = range(node_count,len(interior_nodes)+node_count)
  total_nodes = np.vstack((total_nodes,interior_nodes))
  node_count += len(interior_nodes)
  D = D.buffer(buffer_size)
  T.toc()

  T.tic('Material Properties')
  # Material Properties
  # -------------------
  # load P-wave and S-wave velocities from PREM table
  prem_table = np.loadtxt('PREM.txt',skiprows=1)
  depth = prem_table[:,0] # km
  P_vel = prem_table[:,1]*1e-3 # km/s
  S_vel = prem_table[:,2]*1e-3 # km/s
  rho = prem_table[:,3]*1e6 # kg/km**3

  # lame parameters
  mu = (S_vel**2*rho) # kg/km*s**2
  lam = (P_vel**2*rho - 2*mu) # kg/km*s**2

  # create interpolants. The interpolants are 1D with respect to 
  # depth
  eps = 5*EPS/nearest(depth)[0]
  lam_itp = RBFInterpolant(depth,eps,value=lam)  
  mu_itp = RBFInterpolant(depth,eps,value=mu)  
  rho_itp = RBFInterpolant(depth,eps,value=rho)  
  
  # Convert the 1D interpolants to 2D cartesian interpolants
  r = np.sqrt(np.sum(total_nodes**2,1))
  eps = 5*EPS/nearest(total_nodes)[0]
  lam_itp = RBFInterpolant(total_nodes,eps,value=lam_itp(r))
  mu_itp = RBFInterpolant(total_nodes,eps,value=mu_itp(r))
  rho_itp = RBFInterpolant(total_nodes,eps,value=rho_itp(r))
  T.toc()

  T.tic('Form Problem Matrices')
  # Build Problem Matrices
  # ----------------------
  # calculate shape parameter
  eps = EPS/nearest(total_nodes)[0]

  # Form interpolation, acceleration, and boundary condition matrix
  G = interpolation_matrix(total_nodes,total_nodes,eps)
  A = acceleration_matrix(total_nodes,total_nodes,eps,lam_itp,mu_itp,rho_itp)
  B = traction_matrix(surface_nodes,surface_normals,total_nodes,eps,lam_itp,mu_itp)/1e8
  G[surface_idx,:,:,:] = B
  T.toc()

  # Print Run Information
  # ---------------------
  min_dist = np.min(nearest(total_nodes)[0])
  max_speed = np.max(np.concatenate((P_vel,S_vel)))

  logger.info('minimum distance between nodes (h): %g km' % min_dist)
  logger.info('maximum wavespeed (c): %g km/s' % max_speed)
  logger.info('time step size (t): %g s' % DT)
  logger.info('t*c/h: %g' % (DT*max_speed/min_dist))

  T.tic('Time Stepping')
  # Time Stepping
  # -------------  
  time_steps = np.arange(0.0,TMAX,DT)
  total_steps = len(time_steps)

  # Output Object 
  out_dict = {}  
  out_dict['name'] = name
  out_dict['nodes'] = total_nodes 
  out_dict['surface_nodes'] = surface_nodes 
  out_dict['interior_nodes'] = interior_nodes 
  out_dict['interpolant'] = []
  out_dict['time'] = []
  out_dict['domain'] = D
  out_dict['mu'] = mu_itp
  out_dict['lambda'] = lam_itp
  out_dict['rho'] = rho_itp

  for itr,t in enumerate(time_steps):
    logger.info('starting time step %s of %s' % (itr+1,total_steps))
    if itr == 0:
      # compute initial velocity and displacement
      u,v = initial_conditions(total_nodes) 
      # compute acceleration due to source term 
      f = source(total_nodes,t)/rho_itp(total_nodes)
      # compute acceleration and the spectral coefficients
      a,alpha = F(u,G,A,f,surface_idx,0.0)
      # store solution
      out_dict['interpolant'] += [RBFInterpolant(total_nodes,eps,alpha=alpha,rbf=RBF)]
      out_dict['time'] += [t]
      #u = np.einsum('ijkl,kl',G,alpha)
    else:
      # compute acceleration due to source term 
      f = source(total_nodes,t)/rho_itp(total_nodes)
      # compute next step from leapfrom iteration
      u,v,a,alpha = leapfrog(F,u,v,DT,a=a,F_args=(G,A,f,surface_idx,0.0))
      # store solution
      out_dict['interpolant'] += [RBFInterpolant(total_nodes,eps,alpha=alpha,rbf=RBF)]
      out_dict['time'] += [t]

    if (itr+1)%SAVE_INTERVAL==0: 
      logger.info('saving solution in output/%s' % name)
      f = open('output/%s/%s_%s.pkl' % (name,name,itr),'w')
      pickle.dump(out_dict,f)
      f.close()
      f = open('output/%s/%s.pkl' % (name,name),'w')
      pickle.dump(out_dict,f)
      f.close()

  T.toc()

  # Save Output
  # -----------
  f = open('output/%s/%s.pkl' % (name,name),'w')
  pickle.dump(out_dict,f)
  f.close()

  logger.info('view results with exec/PlotSeisRBF.py %s' % name)

if __name__ == '__main__':
  main()
  
  






