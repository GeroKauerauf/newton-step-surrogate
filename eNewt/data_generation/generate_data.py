# Author: Gero F. Kauerauf

import os
import torch
import numpy as np
from typing import Callable
import h5py

import logging

from ..derivative_wrapper.derivative_wrapper import derivative_wrapper as dw

def hello_world():
  print("hello")

def sample_objective_function(f: Callable[[torch.tensor], torch.tensor], n: int, num_samples: int, subspace: np.array, hdf5_filename: str, append_mode: bool = False):
  """
  Sample objective function
  :param f: objective function
  :param n: input dimension of f
  :param num_samples: number of samples
  :param subspace: subspace to sample from
  """

  # Subspace must define an interval for each dimension
  assert subspace.shape == (2, n), "Subspace must be of shape (2, n)"

  dwf = dw(f)

  logging.info("Sampling objective function")

  x_samples = torch.tensor(np.random.uniform(subspace[0], subspace[1], (num_samples, n)), dtype=torch.float32)

  gradient = torch.vmap(dwf.d1)(x_samples)
  hessian = torch.vmap(dwf.d2)(x_samples)

  logging.info("Storing data to HDF5 file")
  
  # Save to HDF5 file
  # Case distinctions for file handling

  if os.path.exists(hdf5_filename):
    if append_mode:
      # File exists and append mode is enabled. Good!
      logging.info(f"Appending data to existing HDF5 file '{hdf5_filename}'")
      
      with h5py.File(hdf5_filename, "a") as f:
        f["x_samples"].resize((f["x_samples"].shape[0] + x_samples.shape[0]), axis=0)
        f["x_samples"][-x_samples.shape[0]:] = x_samples.numpy()

        f["gradient"].resize((f["gradient"].shape[0] + gradient.shape[0]), axis=0)
        f["gradient"][-gradient.shape[0]:] = gradient.numpy()

        f["hessian"].resize((f["hessian"].shape[0] + hessian.shape[0]), axis=0)
        f["hessian"][-hessian.shape[0]:] = hessian.numpy()

    else:
      # File exists but append mode is disabled. Bad!
      logging.critical(f"File '{hdf5_filename}' already exists and append mode is disabled. Raising Error!")
      raise FileExistsError(f"File '{hdf5_filename}' already exists and append mode is disabled.")
  else:
    # File does not exist. Create new file regardless of append mode. Log warning if append mode is enabled.
    
    if append_mode:
      logging.warning(f"HDF5 File '{hdf5_filename}' does not exist but append mode is enabled. Creating new file!")
    else:
      logging.info(f'Creating new HDF5 file {hdf5_filename}')
    
    with h5py.File(hdf5_filename, "w") as f:
      f.create_dataset("x_samples", data=x_samples.numpy(), maxshape=(None,) + x_samples.shape[1:], chunks=True)
      f.create_dataset("gradient", data=gradient.numpy(), maxshape=(None,) + gradient.shape[1:], chunks=True)
      f.create_dataset("hessian", data=hessian.numpy(), maxshape=(None,) + hessian.shape[1:], chunks=True)

  logging.info("Finished sampling and storing data")
