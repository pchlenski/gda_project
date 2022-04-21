import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import random

def generate_unconfounded(n = 3, linear = True, dropout = 0, N = 1000):
  '''
  Generates SCM with n independent latent variables and 10*n observable variables

  INPUT

  n (int; 3, 4 or 5): number of independent latent variables
  linear (Boolean): whether the SCM functions are all linear
  dropout (float between 0 and 0.999): portion of connections dropped in the weight matrix  
  N (int): number of samples

  OUTPUT

  latent (N x n; float): latent variables
  observables (N x 10*n; float): observable variables
  '''

  if n not in [3,4,5]:
    print('n should be in {3,4,5}')
    return
  if linear not in [True,False]:
    print('linear should be True/False')
    return
  if dropout <0 or dropout >= 1.0:
    print('dropout should be [0,1)')
    return
  if N<=0 or type(N) != int:
    print('N should be a positive integer')

  U = np.zeros((N,n))

  if n ==3:
    U[:,0] = bernoulli.rvs(p=0.3,size=N)
    U[:,1] = bernoulli.rvs(p=0.5,size=N)
    U[:,2] = bernoulli.rvs(p=0.7,size=N) 
    latents = U

    if linear:
      W = np.random.random_sample((n*10*n,1)) # random matrix with weights in [0,1.0)
      total = n*10*n
      dropout = int(dropout*total)
      idx = random.sample(range(total), dropout)
      W[idx] = 0 # drop some connections
      W = np.reshape(W,newshape=(n,10*n))

      observables = latents @ W
  
  return latents, observables


def generate_confounded(n = 3, linear = True, dropout = 0, N = 1000):
  '''
  Generates SCM with n confounded latent variables and 10*n observable variables

  INPUT

  n (int; 3, 4 or 5): number of confounded latent variables
  linear (Boolean): whether the SCM functions are all linear
  dropout (float between 0 and 0.999): portion of connections dropped in the weight matrix  
  N (int): number of samples

  OUTPUT

  latent (N x n; float): latent variables
  observables (N x 10*n; float): observable variables
  '''

  if n not in [3,4,5]:
    print('n should be in {3,4,5}')
    return
  if linear not in [True,False]:
    print('linear should be True/False')
    return
  if dropout <0 or dropout >= 1.0:
    print('dropout should be [0,1)')
    return
  if N<=0 or type(N) != int:
    print('N should be a positive integer')

  U = np.zeros((N,n+1))
  latents = np.zeros((N,n))

  if n ==3:
    U[:,0] = bernoulli.rvs(p=0.3,size=N)
    U[:,1] = bernoulli.rvs(p=0.5,size=N)
    U[:,2] = bernoulli.rvs(p=0.7,size=N) 
    U[:,3] = bernoulli.rvs(p=0.8,size=N) 

    latents[:,0] = (U[:,0] + U[:,3])>0
    latents[:,1] = (U[:,1] + U[:,3])>0
    latents[:,2] = U[:,2]

    if linear:
      W = np.random.random_sample((n*10*n,1)) # random matrix with weights in [0,1.0)
      total = n*10*n
      dropout = int(dropout*total)
      idx = random.sample(range(total), dropout)
      W[idx] = 0 # drop some connections
      W = np.reshape(W,newshape=(n,10*n))

      observables = latents @ W
  
  return latents, observables


def generate_dependent(n = 3, linear = True, dropout = 0, N = 1000):
  '''
  Generates SCM with n dependent latent variables and 10*n observable variables

  INPUT

  n (int; 3, 4 or 5): number of dependent latent variables
  linear (Boolean): whether the SCM functions are all linear
  dropout (float between 0 and 0.999): portion of connections dropped in the weight matrix  
  N (int): number of samples

  OUTPUT

  latent (N x n; float): latent variables
  observables (N x 10*n; float): observable variables
  '''

  if n not in [3,4,5]:
    print('n should be in {3,4,5}')
    return
  if linear not in [True,False]:
    print('linear should be True/False')
    return
  if dropout <0 or dropout >= 1.0:
    print('dropout should be [0,1)')
    return
  if N<=0 or type(N) != int:
    print('N should be a positive integer')

  U = np.zeros((N,n))
  latents = np.zeros((N,n))

  if n ==3:
    U[:,0] = bernoulli.rvs(p=0.3,size=N)
    U[:,1] = bernoulli.rvs(p=0.5,size=N)
    U[:,2] = bernoulli.rvs(p=0.7,size=N) 

    latents[:,0] = U[:,0]
    latents[:,1] = (U[:,1] + latents[:,0])>0
    latents[:,2] = U[:,2]

    if linear:
      W = np.random.random_sample((n*10*n,1)) # random matrix with weights in [0,1.0)
      total = n*10*n
      dropout = int(dropout*total)
      idx = random.sample(range(total), dropout)
      W[idx] = 0 # drop some connections
      W = np.reshape(W,newshape=(n,10*n))

      observables = latents @ W
  
  return latents, observables
