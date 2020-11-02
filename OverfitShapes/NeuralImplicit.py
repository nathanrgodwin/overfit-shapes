import os
import time
from torch import nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from SDFSampler import PointSampler, MeshLoader, normalizeMeshToUnitSphere

class NeuralImplicit:
  def __init__(self, N = 8, H = 32):
    self.model = self.OverfitSDF(N, H)
    self.epochs = 100
    self.lr = 1e-4
    self.batch_size = 64
    self.log_iterations = 1000

  # Supported mesh file formats are .obj and .stl
  # Sampler selects oversample_ratio * num_sample points around the mesh, keeping only num_sample most
  # important points as determined by the importance metric
  def encode(self, mesh_file, num_samples=1000000, oversample_ratio = 30, early_stop=None, verbose=True):
    dataset = self.MeshDataset(mesh_file, num_samples, oversample_ratio)
    dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    self.model.to(device)

    loss_func = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    for e in range(self.epochs):
      epoch_loss = 0
      self.model.train()
      count = 0

      for batch_idx, (x_train, y_train) in enumerate(dataloader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        count += self.batch_size
        optimizer.zero_grad()

        y_pred = self.model(x_train)

        loss = loss_func(y_pred, y_train)

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        if (verbose and ((batch_idx+1) % self.log_iterations == 0)):
          msg = '{}\tEpoch: {}:\t[{}/{}]\tepoch_loss: {:.6f}\tloss: {:.6f}'.format(
              time.ctime(),
              e + 1,
              count,
              len(dataset),
              epoch_loss / (batch_idx + 1),
              loss)
          print(msg)

      if (early_stop and epoch_loss < early_stop):
        break
    
    model_file = os.path.dirname(mesh_file) + "/" + os.path.splitext(os.path.basename(mesh_file))[0] + ".pth"
    torch.save(self.model.state_dict(), model_file)
    
  # The actual network here is just a simple MLP
  class OverfitSDF(nn.Module):
    def __init__(self, N, H):
      super().__init__()
      assert(N > 0)
      assert(H > 0)

      # Original paper uses ReLU but I found this lead to dying ReLU issues
      # with negative coordinates. Perhaps was not an issue with original paper's
      # dataset?
      net = [nn.Linear(3, H), nn.LeakyReLU(0.1)]
      
      for i in range(0,N):
        net += [nn.Linear(H, H), nn.LeakyReLU(0.1)]

      net += [nn.Linear(H,1), nn.LeakyReLU(0.1)]
      self.model = nn.Sequential(*net)

    def forward(self, x):
      x = self.model(x)
      output = torch.tanh(x)
      return output

  # Dataset generates data from the mesh file using the SDFSampler library on CPU
  # Moving data generation to GPU should speed up this process significantly
  class MeshDataset(Dataset):
    def __init__(self, mesh_file, num_samples, oversample_ratio):
      print("Loading " + mesh_file, flush=True)
      time.sleep(0.1)

      vertices, faces = MeshLoader.read(mesh_file)
      normalizeMeshToUnitSphere(vertices, faces)

      print("Loaded mesh", flush=True)
      time.sleep(0.1)

      sampler = PointSampler(vertices, faces)
      self.pts = sampler.sample(num_samples, oversample_ratio)
      print("Sampled " + str(len(self)) + " points", flush=True)
      time.sleep(0.1)

    def __getitem__(self, index):
      return torch.from_numpy(self.pts[0][index,:]), torch.tensor([self.pts[1][index]])

    def __len__(self):
      return self.pts[0].shape[0]
