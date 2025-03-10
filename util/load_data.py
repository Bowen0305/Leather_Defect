import os
import faiss
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from utils.common.embedding import generate_embedding_features
from utils.common.image_processing import PatchMaker, ForwardHook, LastLayerToExtractReachedException
from utils.common.backbones import Backbone

class LeatherDataset(Dataset):
    def __init__(self, transform, gt_transform, phase):
        self.root = 'D:/leather_defect/data/DATASET/'
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
    
    def __len__(self): 
        return len(self.img_paths)
    
    def __getitem__(self , idx):
        path , gt, label, img_type = self.img_paths[idx], self.gt_path[idx], self.labels[idx] , self.types[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        if gt == 0:
            gt = torch.zeros([1, img.size()[-2] , img.size()[-2]])
        
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        return img, gt, label, img_type
    
    def load_dataset(self):

        if self.phase == 'train':
            img_paths = os.listdir(self.root + 'good/')
            
