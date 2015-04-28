#!/usr/bin/env python
#
# User defined functions used by SeisRBF.py. The user should modify
# these functions based on their problem specifications

import numpy as np

def initial_conditions(x):
  '''                            
  Initial displacement an velocity field.

  Parameters
  ----------
    x: array of points where the initial displacments and velocity 
       field are to be output.  This is an (N,D) array where N is 
       the number of points and D is the number of spatial dimensions

  Returns
  -------
    u: displacement field
    v: velocity field               
  '''
  u = np.zeros(np.shape(x)) # returns 0 for initial disp
  v = np.zeros(np.shape(x)) # returns 0 for initial vel
  return u,v


def source_time_function(t):
  '''         
  Describes the time evolution of seismic moment

  Parameters
  ----------
    t: scalar value of time

  Returns
  -------
    The proportion of the seismic moment that has been released at 
    the given time.  This value should be between 0 and 1 
  '''
  out = 1.0 # all seismic moment is instantly released
  return out 


def boundary(t):
  '''   
  parameterized boundary function   
                               
  Parameters         
  ----------             
    t: array with values between 0 and 1     
                                 
  Returns               
  -------               
    The coordinates of the boundary that correspond with the given
    values of t.  The coordinates should make an full loop as t goes
    from 0 to 1 
                             
  '''
  p = 2*np.pi*t
  out = np.zeros((len(t),2))
  out[:,0] =  6371000*np.sin(p) # The boundary is a circle with radius 
  out[:,1] = -6371000*np.cos(p) # equal to that of the earth in meters

  return out

def node_density(x):
  '''                           
  returns a value between 0 and 1 indicating the desired density
  of nodes at the given points. This function corresponds to psi 
  in my paper.      
                                      
  Parameters                         
  ----------                           
    x: Array of points where node density is to be given. This is an
       N by D array, where N is the number of points and D is the 
       number of spatial dimensions
                                                 
  Returns                                             
  -------                                          
    length N array of node densities between 0 and 1       
             
  '''
  layer_depths = [[np.inf,5800],
                  [5800,5000],
                  [5000,3700],
                  [3700,3300],
                  [3300,1500],
                  [1500,1100],
                  [1100,0]]

  layer_depths = 1000*np.array(layer_depths) # meters

  layer_density = [1.0,
                   1.0,
                   0.5,
                   1.0,
                   0.7,
                   0.7,
                   0.7]

  R = np.sqrt(x[0]**2 + x[1]**2)
  #R = np.sqrt(np.sum(x**2,1))
  #out = np.zeros(len(R))
  for depths,density in zip(layer_depths,layer_density):
    if (R<=depths[0]) & (R>depths[1]):
      return density

  #out[(R<=depths[0])&(R>depths[1])] = density     
  #return out

