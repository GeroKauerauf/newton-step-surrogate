# Author: Gero F. Kauerauf

import os
import torch
import numpy as np
from typing import Callable

import logging

from ..derivative_wrapper.derivative_wrapper import derivative_wrapper as dw

def sample_objective_function(f: Callable[[torch.tensor], torch.tensor], n: int, num_samples: int, subspace: np.array, filename: str):
  """
  Sample objective function
  :param f: objective function
  :param n: input dimension of f
  :param num_samples: number of samples
  :param subspace: subspace to sample from
  :param filename: file to store generated data to
  """

  # Subspace must define an interval for each dimension
  assert subspace.shape == (2, n), "Subspace must be of shape (2, n)"

  if os.path.exists(filename):
    logging.critical(f"File '{filename}' already exists!")
    raise FileExistsError(f"File '{filename}' already exists!")
  
  # Wrap function to compute derivatives easily
  dwf = dw(f)

  logging.info("Sampling objective function")

  x_samples = torch.tensor(np.random.uniform(subspace[0], subspace[1], (num_samples, n)), dtype=torch.float32)

  gradient = torch.vmap(dwf.d1)(x_samples)
  hessian = torch.vmap(dwf.d2)(x_samples)

  # Create and save TensorDataset
  dataset = torch.utils.data.TensorDataset(x_samples, gradient, hessian)

  logging.info(f"Creating new file '{filename}'")
  torch.save(dataset, filename)

  logging.info("Finished sampling and storing data")


def load_TensorDataset(filename: str) -> torch.utils.data.TensorDataset:
  """
  Load TensorDataset object from disk
  :para filename: file to load data from
  """

  # Check that file is found
  if not os.path.exists(filename):
    logging.critical(f"File '{filename}' not found!")
    raise FileNotFoundError(f"File '{filename}' not found!")
  
  # Load TensorDataset
  return torch.load(filename)


def get_batches(dataset: torch.utils.data.TensorDataset, batch_size: int):
  if len(dataset) % batch_size != 0:
    logging.critical("Dataset is not divided by batch_size")
    raise ValueError("Dataset is not divided by batch_size")

  num_batches = len(dataset)//batch_size

  return torch.utils.data.random_split(dataset, [batch_size] * num_batches)


def robust_normalization(data_path: str, normalized_data_path: str):
  """
  Robust Normalization of data
  """
  logging.info("Robust Normalization of data")

  if os.path.exists(normalized_data_path):
    logging.critical(f"File '{normalized_data_path}' already exists!")
    raise FileExistsError(f"File '{normalized_data_path}' already exists!")
  if not os.path.exists(data_path):
    logging.critical(f"File '{data_path}' not found!")
    raise FileNotFoundError(f"File '{data_path}' not found!")

  files = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

  medians = []
  IQRs = []

  for file in files:
    dataset = load_TensorDataset(file)

    x, g, h = dataset.tensors
    
    xg_flatten = torch.stack((x.flatten(), g.flatten()))
    h_flatten = h.flatten()

    xg_median = xg_flatten.median()
    h_median = h_flatten.median()
    median = (xg_median + h_median)/2
    medians.append(median)

    xg_IQR = xg_flatten.quantile(0.75) - xg_flatten.quantile(0.25)
    h_IQR = h_flatten.quantile(0.75) - h_flatten.quantile(0.25)
    IQR = (xg_IQR + h_IQR)/2
    IQRs.append(IQR)

  median = torch.stack(medians).mean()
  IQR = torch.stack(IQRs).mean()

  logging.info(f"Computed median={median}, IQR={IQR} of data in {data_path}")

  logging.info("Creating new directory for normalized data")
  os.makedirs(normalized_data_path)

  for file in files:
    dataset = load_TensorDataset(file)

    x, g, h = dataset.tensors

    # g_normalized = (g - median)/IQR
    # h_normalized = (h - median)/IQR
    g_normalized = g/IQR
    h_normalized = h/IQR


    dataset_normalized = torch.utils.data.TensorDataset(x, g_normalized, h_normalized)

    filename = os.path.join(normalized_data_path, "normalized_"+os.path.basename(file))
    torch.save(dataset_normalized, filename)


  logging.info("Finished Robust Normalization of data")
