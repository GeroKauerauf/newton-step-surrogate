# Author: Gero F. Kauerauf

import torch

def rosenbrock(x: torch.tensor) -> torch.tensor:
  """
  Rosenbrock function
  :param x: input tensor
  :return: output tensor
  """
  return torch.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
