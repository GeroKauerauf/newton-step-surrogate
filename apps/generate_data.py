# Author: Gero F. Kauerauf

import sys, os.path
# add relative path to sys.path
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import numpy as np



import eNewt
from objective_functions.objective_function import rosenbrock

eNewt.sample_objective_function(f=rosenbrock, n=3, num_samples=7, subspace=np.array([[0, 0, 0], [1, 0, 1]]), hdf5_filename='data.h5', append_mode=True)
