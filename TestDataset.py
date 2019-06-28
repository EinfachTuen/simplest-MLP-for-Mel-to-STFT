import multiprocessing

from torch.utils.data import Dataset
import numpy as np

class TestDataset(Dataset):
    def __init__(self, test_file):
        self.test_file = test_file
        self.data = []

    def initialize(self):
        self.data.append(self.load_files())

    def load_files(self):
        input_file_name = self.test_file
        input = np.load(input_file_name)
        return input

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
