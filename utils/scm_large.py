# importing necessary libraries
try:
    import pycid
except ModuleNotFoundError:
    import sys

    !{sys.executable} -m pip install git+https://github.com/causalincentives/pycid # for the latest development version
    import pycid

import numpy as np
import random
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from gda_project.utils.embed import *
import warnings
warnings.filterwarnings('ignore')

def print_scm_large(index=1):
  if index == 1:
    cbn = pycid.CausalBayesianNetwork(
        [("U1", "A"), ("U1", "B"), ("U5","C"), ("U2","D"), ("U2","E"), ("U2","F"),
        ("U3","F"), ("U3","G"), ("U4","H"), ("U4","I")])
    cbn.draw()
    
 def scm_large(index=1, N=1000, noise=0.0, dropout=0.0):

  if index == 1:
    # define first SCM
    cbn = pycid.CausalBayesianNetwork(
        [("U1", "A"), ("U1", "B"), ("U5","C"), ("U2","D"), ("U2","E"), ("U2","F"),
        ("U3","F"), ("U3","G"), ("U4","H"), ("U4","I")])
    for node in ["U1","U2","U3","U4","U5"]:
      cardinality = np.random.randint(2,10)
      support = random.sample(range(100),cardinality)
      cbn.model[node] = pycid.discrete_uniform(support)
    cbn.model["A"] = lambda U1: 5*U1 + 6
    cbn.model["B"] = lambda U1: 10*U1 + 3
    cbn.model["C"] = lambda U5: U5
    cbn.model["D"] = lambda U2: 7*U2 + 4
    cbn.model["E"] = lambda U2: 3*U2 + 5
    cbn.model["F"] = lambda U2,U3: 4*U2 + 6*U3 + 3
    cbn.model["G"] = lambda U3: 9*U3 + 1
    cbn.model["H"] = lambda U4: 13*U4 + 2
    cbn.model["I"] = lambda U4: 2*U4 + 11
  else:
    print("ERROR: enter index value in 1-3")
    return

  # generate 9-latent samples
  latents = np.zeros((N,9))
  for i in tqdm(range(N)):
    latents[i] = np.array(list(cbn.sample().values())[:9])
  
  # generate 100-observable samples
  n_observables = 100
  W = np.random.random_sample((9*n_observables,1)) # random matrix with weights in [0,1.0)
  total = 9*n_observables
  dropout = int(dropout*total)
  # print("check3")
  idx = random.sample(range(total), dropout)
  W[idx] = 0 # drop some connections
  W = np.reshape(W,newshape=(9,n_observables))
  observables = latents @ W
  observables += np.random.normal(0,noise,n_observables) # add noise
  
  # generate regression labels
  vec = np.random.randint(1,20,9)
  threshold = np.random.randint(2*9 + 1)
  Y = np.zeros((N,9))
  # Y = (latents @ vec.T) > threshold
  Y = (latents @ vec.T)
  
  return latents, observables, Y
