import numpy as np
import torch
import glob
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

class PrivateDataset(Dataset):
    def __init__(self, root, transform , gt_transform, phase):
        self.root = 'D:/leather_defect/data/DATASET/'
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()
    
    def load_dataset(self):
        img_paths = []
        gt_paths = []
        labels = [] # 0:good, 1: black, 2: wrinkle
        
        for label , i in enumerate (['good' , 'black', 'wrinkle']):
            path = glob.glob(self.root + label + "/*.jpg")
            labels.append([label] * len(path))
            if label != 'good':
                gt_path = glob.glob(self.root + 'ground_truth/' + label + "/*.jpg")
            else:
                gt_path = [None] * len(path)
            img_paths.append(path)
            gt_paths.append(gt_path)
        
        return img_paths, gt_paths, labels
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self , i):
        img_path, gt_path, label = self.img_paths[i] , self.gt_paths[i], self.labels[i]
        img = Image.open(img_path).convert('RGB')
        img = Standardize(img)
        if gt_path == None:
            gt_img = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt_img = Transform(Image.open(gt_path))
        
        return img, gt_img, os.path.basename(img_path[:-4]), label