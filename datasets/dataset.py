import torch
from torch.utils.data import Dataset
import os
import glob
import cv2

class MNISTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Initialize your dataset here
        self.data_dir = data_dir
        self.file_ls = glob.glob(os.path.join(data_dir, "*/**"))

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.file_ls)

    @staticmethod
    def spad(x, exp=0.4):
        probs = 1-np.exp(-x*exp)
        return np.random.binomial(n=1, p=probs)

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        inp = cv2.imread(self.file_ls[idx])[:,:,0]/255
        inp = np.pad(inp, pad_width=2)
        return {'input':inp, 'gt':spad(inp)}
