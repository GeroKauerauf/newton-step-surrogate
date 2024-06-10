# Author: Gero F. Kauerauf

import torch

class Factorized_Hessian_Model(torch.nn.Module):
  """
  A model that approximates the gradient and a factorized Hessian of a function mapping n inputs to 1 output.
  :param n: The number of inputs to the function.
  """
  
  _n: int
  _backbone: torch.nn.ModuleList
  _gradient_head: torch.nn.ModuleList
  _hessian_head: torch.nn.ModuleList

  def __init__(self, n: int):
    super(Factorized_Hessian_Model, self).__init__()
    self._n = n

    self._backbone = torch.nn.ModuleList()
    self._gradient_head = torch.nn.ModuleList()
    self._hessian_head = torch.nn.ModuleList()

    self._backbone.append(torch.nn.Linear(n, 100*n*n))
    for i in range(3):
      self._backbone.append(torch.nn.Linear(100*n*n, 100*n*n))
    
    self._gradient_head.append(torch.nn.Linear(100*n*n, 100*n))
    for i in range(4):
      self._gradient_head.append(torch.nn.Linear(100*n, 100*n))
    self._gradient_head.append(torch.nn.Linear(100*n, n))

    self._hessian_head.append(torch.nn.Linear(100*n*n, 100*n*n))
    for i in range(4):
      self._hessian_head.append(torch.nn.Linear(100*n*n, 100*n*n))
    self._hessian_head.append(torch.nn.Linear(100*n*n, n*n))


  def forward(self, x: torch.tensor):
    activation = torch.nn.SiLU()
    for layer in self._backbone:
      x = activation(layer(x))
    
    g = x
    h = x
    
    for layer in self._gradient_head[:-1]:
      g = activation(layer(g))
    g = self._gradient_head[-1](g)

    for layer in self._hessian_head[:-1]:
      h = activation(layer(h))
    h = self._hessian_head[-1](h).view(-1, self._n, self._n)

    return g, h


  def newton_step(self, x: torch.tensor, median, IQR, alpha: float=1.0):
    g, h = self(x)
    L = torch.tril(h, diagonal=-1) # set unitriangular=True in solver!
    U = torch.triu(h) # *IQR
    
    g.unsqueeze_(1)
    # g = g # *IQR
    
    print("cond(L)=", torch.linalg.cond(L+torch.eye(self._n)))
    print("cond(U)=", torch.linalg.cond(U))

    Z = torch.linalg.solve_triangular(L, -g, upper=False, unitriangular=True)
    delta_x = torch.linalg.solve_triangular(U, Z, upper=True).squeeze()
    # print("LU delta_x=", delta_x)
    # H = (L+torch.eye(self._n)).squeeze()@U.squeeze()
    # print("cond(H)=", torch.linalg.cond(H))

    # g = g.squeeze()
    # delta_x = torch.linalg.inv(H)@(-g)

    # print("h=",h)
    # print("g=",g)

    # delta_x = torch.linalg.inv(h)@g
    
    print("delta_x=", delta_x)
    
    return x + alpha*delta_x
