from torch.utils.data import Dataset
import torch
import random
import pandas as pd
from rdkit import Chem
import pickle
from rdkit import RDLogger
from calc_property import calculate_property
RDLogger.DisableLog('rdApp.*')

class SMILESDataset_SHIN_MLM(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False, test=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.test = test

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['SMILES'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

        if self.test:
            return '[CLS]' + smiles

        value = torch.tensor(self.data[index]['MLM'].item())
        return '[CLS]' + smiles, value

class SMILESDataset_SHIN_HLM(Dataset):
    def __init__(self, data_path,data_length=None, shuffle=False, test=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.test = test

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['SMILES'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

        if self.test:
            return '[CLS]' + smiles

        value = torch.tensor(self.data[index]['HLM'].item())
        return '[CLS]' + smiles, value

