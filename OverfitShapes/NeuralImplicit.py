import os
import sys
import logging
import time
from torch import nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from OverfitShapes import PointSampler, MeshLoader, normalizeMeshToUnitSphere

class NeuralImplicit:
  def __init__(self, H = 8, N = 32):
    self.N = N
    self.H = H
    self.model = self.OverfitSDF(H, N)
    self.epochs = 100
    self.lr = 1e-4
    self.batch_size = 64
    self.log_iterations = 1000
    self.boundary_ratio = 0.95
    self.trained = False
    self.adaptive_lr = False

  # Supported mesh file formats are .obj and .stl
  # Sampler selects oversample_ratio * num_sample points around the mesh, keeping only num_sample most
  # important points as determined by the importance metric
  def encode(self, mesh_file, num_samples=1000000, oversample_ratio = 10, early_stop=None, verbose=True):
    if (verbose and not logging.getLogger().hasHandlers()):
      logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
      logging.getLogger().setLevel(logging.INFO)

    mesh_basename = os.path.basename(mesh_file)
    dataset = self.MeshDataset(mesh_file, num_samples, oversample_ratio, self.boundary_ratio, verbose)
    dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    self.model.to(device)

    loss_func = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    scheduler = None
    if (self.adaptive_lr):
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                    patience=min(self.epochs/20, 10), verbose=verbose)


    for e in range(self.epochs):
      epoch_loss = 0
      self.model.train(True)
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
          msg = '{}\tEpoch: {}:\t[{}/{}]\tepoch_loss: {:.6f}'.format(
              mesh_basename,
              e + 1,
              count,
              len(dataset),
              epoch_loss / (batch_idx + 1))
          logging.info(msg)

      if (scheduler is not None):
        scheduler.step(epoch_loss)

      if (early_stop and epoch_loss < early_stop):
        break
  
    model_file = os.path.dirname(os.path.abspath(mesh_file)) + "/" + os.path.splitext(os.path.basename(mesh_file))[0] + ".pth"
    torch.save(self.model.state_dict(), model_file)
    self.trained = True

  def load(self, state_file):
    try:
      self.model.load_state_dict(torch.load(state_file))
      self.trained = True
      return True
    except Exception as e:
      print("Failed to load " + state_file)
      return False

  # Returns weights in row major form
  def weights(self):
    self.model.to(torch.device("cpu"))
    weights = np.empty((0,))
    for weight_mat in list(self.model.state_dict().values())[::2]:
      weights = np.concatenate((weights, np.squeeze(weight_mat.numpy().reshape(-1, 1))))
    return weights

  # Returns biases in row major form
  def biases(self):
    self.model.to(torch.device("cpu"))
    biases = np.empty((0,))
    for bias_mat in list(self.model.state_dict().values())[1::2]:
      biases = np.concatenate((biases, bias_mat.numpy()))
    return biases

  def renderable(self):
    return (self.H, self.N, self.weights(), self.biases())

  # The actual network here is just a simple MLP
  class OverfitSDF(nn.Module):
    def __init__(self, H, N):
      super().__init__()
      assert(N > 0)
      assert(H > 0)

      # Original paper uses ReLU but I found this lead to dying ReLU issues
      # with negative coordinates. Perhaps was not an issue with original paper's
      # dataset?
      net = [nn.Linear(3, N), nn.LeakyReLU(0.1)]
      
      for _ in range(H-1):
        net += [nn.Linear(N, N), nn.LeakyReLU(0.1)]

      net += [nn.Linear(N,1)]
      self.model = nn.Sequential(*net)

    def forward(self, x):
      x = self.model(x)
      output = torch.tanh(x)
      return output

  # Dataset generates data from the mesh file using the SDFSampler library on CPU
  # Moving data generation to GPU should speed up this process significantly
  class MeshDataset(Dataset):
    def __init__(self, mesh_file, num_samples, oversample_ratio, boundary_ratio = 0.99, verbose=True):
      if (verbose):
        logging.info("Loading " + mesh_file)

      vertices, faces = MeshLoader.read(mesh_file)
      normalizeMeshToUnitSphere(vertices, faces)

      if (verbose):
        logging.info("Loaded " + mesh_file)

      sampler = PointSampler(vertices, faces)
      boundary_points = sampler.sample(int(boundary_ratio*num_samples), oversample_ratio)

      # Testing indicated very poor SDF accuracy outside the mesh boundary which complicated
      # raymarching operations.
      # Adding samples through the unit sphere improves accuracy farther from the boundary,
      # but still within the unit sphere
      general_points = sampler.sample(int((1-boundary_ratio)*num_samples), 1)
      self.pts = (np.concatenate((boundary_points[0], general_points[0])),
                  np.concatenate((boundary_points[1], general_points[1])))
        
      if (verbose):
        logging.info("Sampled " + str(len(self)) + " points: " + mesh_file)

    def __getitem__(self, index):
      return torch.from_numpy(self.pts[0][index,:]), torch.tensor([self.pts[1][index]])

    def __len__(self):
      return self.pts[0].shape[0]