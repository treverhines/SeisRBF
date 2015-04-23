#!/usr/bin/env python
from __future__ import division
import numpy as np

class Prime(object):
  '''
  enumerates over N prime numbers
  '''  
  def __init__(self,N):
    self.N = N
    self.n = 0

  def __iter__(self):
    return self

  def next(self):
    if self.n == self.N:
      raise StopIteration

    elif self.n == 0:
      self.primes = [2]
      self.n += 1

    else:
      test = self.primes[-1] + 1
      while True:
        if not any(test/p == test//p for p in self.primes):
          self.primes += [test]  
          self.n += 1
          break 

        test += 1

    out = self.primes[-1]
    return out

class Halton(object):
  '''
  A class which produces a Halton sequence when called and 
  remembers the state of the sequence so that repeated calls
  produce the next items in the sequence.
  ''' 
  def __init__(self,dim=1,start=0,skip=1):
    self.count = start
    self.skip = skip
    self.dim = dim

  def __call__(self,N):
    out = halton(N,self.dim,self.count,self.skip)
    self.count += N*self.skip    
    return out

  def qunif(self,low=0.0,high=1.0,N=50):
    return self(N)*(high-low) + low

  def qnorm(self,mu=0.0,sigma=1.0,N=50):
    self.dim *= 2
    unif = self(N)
    unif1 = unif[:,:self.dim/2]
    unif2 = unif[:,self.dim/2:]
    out = np.sqrt(-2*np.log(unif1))*np.cos(2*np.pi*unif2)
    self.dim /= 2
    return out*sigma + mu

def halton(N,dim=1,start=0,skip=1):
  '''
  returns a halton sequence using the algorithm 
  from http://en.wikipedia.org/wiki/Halton_sequence
  '''
  out = np.zeros((N,dim))
  for d,base in enumerate(Prime(dim)):
    i = start + 1 + np.arange(0,skip*N,skip)
    f = np.ones(N)
    while any(i > 0):
      f = f/base
      out[:,d] += f*(i%base)
      i //= base 
    
  return out

def qunif(low=0.0,high=1.0,N=50,dim=1,start=0,skip=1):
  '''
  quasi-uniform random number generator
  '''
  return halton(N,dim,start,skip)*(high-low) + low
  
def qnorm(mu=0.0,sigma=1.0,N=50,dim=1,start=0,skip=1):
  ''' 
  quasi-normal random number generator
  '''
  unif = halton(N,2*dim,start,skip)
  unif1 = unif[:,:dim]
  unif2 = unif[:,dim:]
  out = np.sqrt(-2*np.log(unif1))*np.cos(2*np.pi*unif2)
  return out*sigma + mu


      
    
  







    
    
