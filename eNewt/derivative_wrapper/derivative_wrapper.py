# Author: Gero F. Kauerauf

import torch

from typing import Callable

class derivative_wrapper():
  def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]):
    """
    :param f: function $f:R^n\to R$ to be differentiated
    """
    self._f = f
    self._d1 = torch.func.jacrev(f)
    self._d2 = torch.func.hessian(f)

  def d1(self, x: torch.Tensor) -> torch.Tensor:
    """
    :param x: input tensor
    :return: first derivative of f evaluated at x
    """
    return self._d1(x)
  def d2(self, x: torch.Tensor) -> torch.Tensor:
    """
    :param x: input tensor
    :return: second derivative of f evaluated at x
    """
    return self._d2(x)
