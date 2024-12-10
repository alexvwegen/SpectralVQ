import os
import numpy as np
import torch
from torch.utils.data import Dataset


class AudioFeatureData(Dataset):
    """
    Dataset for loading precomputed audio features from npy files.
    """
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        feature_path = os.path.join(self.folder, file)
        feature = np.load(feature_path)

        return torch.from_numpy(feature)
    

def take_slice(batch, num_frames=60):
    sliced_samples = [sample[:,:num_frames] for sample in batch]
    return torch.stack(sliced_samples)