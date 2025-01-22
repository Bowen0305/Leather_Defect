import faiss
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, random_split

class PrivateDataset(Dataset):
    def __init__(self, args, root, transform , gt_transform, phase):
        if phase == 'train':
            self.img_path = 