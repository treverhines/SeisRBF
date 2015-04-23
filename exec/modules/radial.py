#!/usr/bin/env python
from __future__ import division
import sympy 
import numpy as np

_R = sympy.symbols('R')
_EPS = sympy.symbols('EPS')

class RBF(object):
  def __init__(self,expr):    
    assert expr.has(_R), (
      'RBF expression does not contain _R')
    
    assert expr.has(_EPS), (
      'RBF expression does not contain _EPS')
    self.R_expr = expr
    self.diff_dict = {}

  def __call__(self,x,c,eps=None,diff=None):
    '''
    D: number of dimensions
    N: number of sample points
    M: numer of radii
    
    x: shape (N,) or (N,D)
    c: shape (M,) or (M,D)
    out: shape (N,M)

    shapes get converted to (N,M,D) arrays

    '''
    x = np.asarray(x)
    c = np.asarray(c)   
    xshape = np.shape(x)
    cshape = np.shape(c)
    assert (len(xshape) == 1) | (len(xshape) == 2), (
      'x must be a 1-D or 2-D array')
    assert (len(cshape) == 1) | (len(cshape) == 2), (
      'c must be a 1-D or 2-D array')

    if len(xshape) == 1:
      x = x[:,None,None]

    if len(cshape) == 1:
      c = c[None,:,None]

    if len(xshape) == 2:
      x = x[:,None,:]

    if len(cshape) == 2:
      c = c[None,:,:]

    N = np.shape(x)[0]
    M = np.shape(c)[1]
    assert np.shape(x)[2] == np.shape(c)[2], (
      'if x and c are 2-D arrays then their second dimensions must have the '
      'same length')
    dim = np.shape(x)[2]
    if eps is None:
      eps = np.ones(M)

    eps = np.asarray(eps)
    assert len(np.shape(eps)) == 1, (
      'eps must be a 1D array')

    assert len(eps) == M, (
      'length of eps must be equal to the number of centers')

    x = np.einsum('ijk->kij',x)
    c = np.einsum('ijk->kij',c)

    if diff is None:
      diff = (0,)*dim

    while len(diff) < dim:
      diff += (0,)

    assert len(diff) == dim, (
      'cannot specify derivatives for dimensions that are higher than the '
      'dimensions of x and center')

    if diff not in self.diff_dict:
      self._make_function(diff)

    args = (tuple(x)+tuple(c)+(eps,))    
    return self.diff_dict[diff](*args)

  def _make_function(self,diff):
    dim = len(diff)
    c_sym = sympy.symbols('c:%s' % dim)
    x_sym = sympy.symbols('x:%s' % dim)    
    r = sympy.sqrt(sum((x_sym[i]-c_sym[i])**2 for i in range(dim)))
    expr = self.R_expr.subs(_R,r)            
    for direction,order in enumerate(diff):
      if order == 0:
        continue
      expr = expr.diff(*(x_sym[direction],)*order)

    self.diff_dict[diff] = sympy.lambdify(x_sym+c_sym+(_EPS,),expr,'numpy')

class RBFInterpolant(object):
  '''
  An RBF interpolant for a given set of collocation points and values

    D: number of domain dimensions
    R: number of range dimensions
    N: number of collocation points
    
    x: (N,) or (N,D)
    eps: (N,)
    value: (N,) or (N,R) 
    alpha: (N,) or (N,R)

    output: (N,R)

    if D or R are not given then they are assumed 1
  '''
  def __init__(self,
               x,
               eps, 
               value=None,
               alpha=None,
               rbf=None):
    if rbf is None:
      rbf = mq

    assert (value is not None) != (alpha is not None), (
      'either val or alpha must be given')

    x = np.asarray(x)
    eps = np.asarray(eps)
    x_shape = np.shape(x)
    N = x_shape[0]
    assert len(x_shape) <= 2
    assert np.shape(eps) == (N,)

    if len(x_shape) == 1:
      x = x[:,None]

    if alpha is not None:
      alpha = np.asarray(alpha)
      alpha_shape = np.shape(alpha)
      assert len(alpha_shape) <= 2
      assert alpha_shape[0] == N
      if len(alpha_shape) == 1:
        alpha = alpha[:,None]

      alpha_shape = np.shape(alpha)
      R = alpha_shape[1]
   
    if value is not None:
      value = np.asarray(value)
      value_shape = np.shape(value)
      assert len(value_shape) <= 2
      assert value_shape[0] == N
      if len(value_shape) == 1:
        value = value[:,None]

      value_shape = np.shape(value)
      R = value_shape[1]

    if alpha is None:
      alpha = np.zeros((N,R))
      G = rbf(x,x,eps)
      for r in range(R):
        alpha[:,r] = np.linalg.solve(G,value[:,r])

    self.x = x
    self.eps = eps
    self.alpha = alpha
    self.R = R
    self.rbf = rbf

  def __call__(self,xitp,diff=None):
    out = np.zeros((len(xitp),self.R))
    for r in range(self.R):
      out[:,r] = np.sum(self.rbf(xitp,
                                 self.x,
                                 self.eps,
                                 diff=diff)*self.alpha[:,r],1)

    return out 



_FUNCTION_DOC = '''
  evaluates M radial basis functions (RBFs) with arbitary dimension at N points.

  Parameters                                       
  ----------                                         
    x: ((N,) or (N,D) array) D dimensional locations to evaluate the RBF
                                                                          
    centers: ((M,) or (M,D) array) D dimensional centers of each RBF
                                                                 
    eps: ((M,) array, default=np.ones(M)) Scale parameter for each RBF
                                                                           
    diff: ((D,) tuple, default=(0,)*dim) a tuple whos length is equal to the number 
      of spatial dimensions.  Each value in the tuple must be an integer
      indicating the order of the derivative in that spatial dimension.  For 
      example, if the the spatial dimensions of the problem are 3 then 
      diff=(2,0,1) would compute the second derivative in the first dimension
      and the first derivative in the third dimension.

  Returns
  -------
    out: (N,M) array for each M RBF evaluated at the N points

  Note
  ----
    the derivatives are computed symbolically in Sympy and then lambdified to 
    evaluate the expression with the provided values.  The lambdified functions
    are cached in the scope of the radial module and will be recalled if 
    a value for diff is used more than once in the Python session.        
'''

_IQ = RBF(1/(1+(_EPS*_R)**2))
def iq(*args,**kwargs):
  '''                                                                                                            
  Inverse Quadratic
  '''                                                             
  return _IQ(*args,**kwargs)

iq.__doc__ += _FUNCTION_DOC

_GA = RBF(sympy.exp(-(_EPS*_R)**2))
def ga(*args,**kwargs):
  '''                                                                                                            
  Gaussian
  '''
  return _GA(*args,**kwargs)

ga.__doc__ += _FUNCTION_DOC

_MQ = RBF(sympy.sqrt(1 + (_EPS*_R)**2))
def mq(*args,**kwargs):
  '''                                                                                                            
  multiquadratic
  '''
  return _MQ(*args,**kwargs)

mq.__doc__ += _FUNCTION_DOC


