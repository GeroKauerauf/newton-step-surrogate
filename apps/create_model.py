# Author: Gero F. Kauerauf

import sys
import os.path
# add relative path to sys.path
sys.path.append(os.path.dirname(sys.path[0]))

import torch

import eNewt

from objective_functions.objective_function import rosenbrock

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = eNewt.models.factorized_hessian_model.Factorized_Hessian_Model(n=2)


# train model
train_data_path = "train_data_normalized/"
# train_data_path = "train_data/"

if not os.path.exists(train_data_path):
  raise FileNotFoundError("No training data found at " + train_data_path)

filenames = [os.path.join(train_data_path, f) for f in os.listdir(train_data_path) if os.path.isfile(os.path.join(train_data_path, f))]


def move_dataset_to_device(dataset, device):
    tensors = [tensor.to(device) for tensor in dataset.tensors]
    return torch.utils.data.TensorDataset(*tensors)


datasets = [move_dataset_to_device(torch.load(f), device) for f in filenames]

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(150):
  for dataset in datasets:
    batches = eNewt.utils.data.get_batches(dataset, 1000)
    data_loaders = [
      torch.utils.data.DataLoader(
        batch, batch_size=1000, shuffle=True, pin_memory=True
      )
      for batch in batches
    ]

    for data_loader in data_loaders:
      for x, g, h in data_loader:
        optimizer.zero_grad()

        g_pred, h_pred = model(x)
        
        L = torch.eye(model._n) + torch.tril(h_pred, diagonal=-1)
        U = torch.triu(h_pred)
        kappa = torch.linalg.cond(L)*torch.linalg.cond(U)

        loss = criterion(g_pred, g) + criterion(L@U, h) # + 0.001*kappa.mean() 
        loss.backward()
        
        optimizer.step()
  print("Epoch", epoch, "loss", loss.item())

x = torch.tensor([[1.0, 1.0], [-1.5, 0.5]])
g, h = model(x)

g_true = torch.vmap(eNewt.derivative_wrapper.derivative_wrapper.derivative_wrapper(rosenbrock).d1)(x) # what the fuck
h_true = torch.vmap(eNewt.derivative_wrapper.derivative_wrapper.derivative_wrapper(rosenbrock).d2)(x)

median = 100.18222045898438
IQR = 332.68389892578125

print("g_true=", g_true)
print("g_pred=", g*IQR)

print("h_true=", h_true)
print("h_pred=", ((torch.eye(model._n)+torch.tril(h,diagonal=-1))@torch.triu(h))*IQR)


# Surrogate Newton steps
print()
print("Surrogate Newton")
print()

with torch.no_grad():
  x = torch.tensor([-1.5, 0.5])
  print("x_start=", x)
  for _ in range(30):
    x = model.newton_step(x=x, median=median, IQR=IQR, alpha=0.5)
    print("x=",x)

    print()

# Exact Newton Steps
print()
print("Exact Newton")
print()

x = torch.tensor([-1.5, 0.5])
print("x_start=", x)
for _ in range(30):
  g = eNewt.derivative_wrapper.derivative_wrapper.derivative_wrapper(rosenbrock).d1(x)
  h = eNewt.derivative_wrapper.derivative_wrapper.derivative_wrapper(rosenbrock).d2(x)

  # print("h=",h)
  # print("g=",g)

  # Breaks newton
  # g = (g-median)/IQR
  # h = (h-median)/IQR

  delta_x = -torch.linalg.inv(h)@g
  print("delta_x=",delta_x)

  x = x + 0.5*delta_x
  print("x=", x)

  print()
