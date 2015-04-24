#!/usr/bin/env python
''' 
user defined functions for SeisRBF.py
'''
import numpy as np
from shapely.geometry import LineString

def initial_conditions(x):
  '''                            
  Initial displacement an velocity field.          
  '''
  N = len(x)
  u = np.zeros(np.shape(x))
  v = np.zeros(np.shape(x))
  return u,v


def source_time_function(t):
  '''         
  a function describing the time history of the force 

  This function should integrate to 1 on the interval 0 to inf     
  '''
  if t < 20.0:
    return 0.05

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
  x =  6371*np.sin(p)
  y = -6371*np.cos(p)
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
  layer_depths = [[np.inf,5800],
                  [5800,5400],
                  [5400,3500],
                  [3500,3100],
                  [3100,1200],
                  [1200,800],
                  [800,0]]

  #layer_depths = [[np.inf,5600],
  #                [5600,3300],
  #                [3300,1000],
  #                [1000,0000]]

  layer_density = [1.0,
                   1.0,
                   0.6,
                   1.0,
                   0.4,
                   0.4,
                   0.4]


  R = np.sqrt(np.sum(x**2,1))
  out = np.zeros(len(R))
  for depths,density in zip(layer_depths,layer_density):
    out[(R<=depths[0])&(R>depths[1])] = density

  return out

