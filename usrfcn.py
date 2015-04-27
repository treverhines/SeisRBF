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
  sig = 2000.0
  N = len(x)
  u = np.zeros(np.shape(x))
  #c = (x[:,0]**2 + x[:,1]**2)
  #a = 1.0/(2*sig**2*np.pi)
  #r = np.sqrt(x[:,0]**2 + x[:,1]**2)
  #theta = np.arctan2(x[:,1],x[:,0])
  # compressional initial
  #u[:,0] = np.exp(-(r/sig)**2)*np.cos(theta)
  #u[:,1] = np.exp(-(r/sig)**2)*np.sin(theta)
  #u[:,0] = np.exp(-(r/sig)**2)*np.cos(theta+np.pi/2)
  #u[:,1] = np.exp(-(r/sig)**2)*np.sin(theta+np.pi/2)
  v = np.zeros(np.shape(x))
  return u,v

def source_time_function(t):
  '''         
  a function describing the time history of the force 

  This function should be 1 as t goes to inf     
  '''
  if t < 2000.0:
    return 1.0

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
  x =  6371000*np.sin(p) # meters
  y = -6371000*np.cos(p) # meters
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


  R = np.sqrt(np.sum(x**2,1))
  out = np.zeros(len(R))
  for depths,density in zip(layer_depths,layer_density):
    out[(R<=depths[0])&(R>depths[1])] = density

  return out

