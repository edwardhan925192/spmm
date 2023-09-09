from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import pickle
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class FEATURE_train_dataset(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False, test=False):
      data = pd.read_csv(data_path)
      self.data = [data.iloc[i] for i in range(len(data))]

      self.test = test

      if shuffle: random.shuffle(self.data)
      if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
      row = self.data.iloc[index]

      # Convert all columns except the first one to tensors
      tensor_row = torch.tensor(row.values[1:], dtype=torch.float)

      return tensor_row

class FEATURE_test_dataset(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
      data = pd.read_csv(data_path)
      self.data = [data.iloc[i] for i in range(len(data))]

      if shuffle: random.shuffle(self.data)
      if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
      row = self.data.iloc[index]

      # Convert all columns except the first one to tensors
      tensor_row = torch.tensor(row.values, dtype=torch.float)

      return tensor_row
