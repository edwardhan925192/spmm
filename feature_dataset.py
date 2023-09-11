from torch.utils.data import Dataset
import torch
import random
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class FEATUREDataset(Dataset):
    def __init__(self, data_path, mode='train', fold_num=0, shuffle=False):
        assert mode in ['train', 'val', 'test'], "Mode should be either 'train', 'val', or 'test'"
        
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]
        
        # Defining validation index ranges
        self.validation_ranges = [(0, 400), (800, 1200), (1600, 2000), (2400, 2800), (3000, 3400)]
        
        if mode == 'val':
            val_start, val_end = self.validation_ranges[fold_num]
            self.current_data = self.data[val_start:val_end]
        elif mode == 'train':
            val_start, val_end = self.validation_ranges[fold_num]
            self.current_data = self.data[:val_start] + self.data[val_end:]
        elif mode == 'test':
            # For test mode, use the entire dataset
            self.current_data = self.data

        if shuffle:
            random.shuffle(self.current_data)

        self.mode = mode

    def __len__(self):
        return len(self.current_data)

    def __getitem__(self, index):        
        row = self.current_data[index]
        
        # For train and validation, exclude the first column
        # For test, include all columns
        values_to_convert = row.values[1:] if self.mode != 'test' else row.values
        
        tensor_row = torch.tensor(values_to_convert, dtype=torch.float)

        return tensor_row
