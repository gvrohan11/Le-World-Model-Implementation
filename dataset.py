import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

'''
Build the data loader - this reads real (frame, next-frame, action) examples from the dataset
and hands them to the model in batches, replacing the fake torch.randn data
'''

class SO100Pairs(Dataset):
    def __init__(self, h5_path, camera="pixels_top"):
        self.h5_path = h5_path
        self.camera = camera
        self.file = None
        with h5py.File(h5_path, "r") as f:
            ep = f["episode_index"][:]
        self.valid = np.where(ep[:-1] == ep[1:][0])[0]

    def __len__(self):
        return len(self.valid)

    def _f(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
        return self.file