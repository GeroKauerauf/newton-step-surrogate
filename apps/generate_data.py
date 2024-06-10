# Author: Gero F. Kauerauf

import sys
import os.path
# add relative path to sys.path
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import numpy as np



import eNewt
from objective_functions.objective_function import rosenbrock

# eNewt.utils.data.sample_objective_function(f=rosenbrock, n=2, num_samples=1000, subspace=np.array([[-2, -1], [2, 3]]), filename='train_data/data-20.pt')

# dataset = eNewt.utils.data.load_TensorDataset('data.pt')

# for batch in eNewt.utils.data.get_batches(dataset, 2):
#   for x, gradient, hessian in batch:
#     print("x=", x)

eNewt.utils.data.robust_normalization("train_data/", "train_data_normalized/")
