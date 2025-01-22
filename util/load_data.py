import faiss
import numpy as np
import torch
import glob
import os


from torch.utils.data import Dataset, DataLoader, random_split

class PrivateDataset(Dataset):
    def __init__(self, args, root, transform , gt_transform, phase):
        self.root = 'D:/leather_defect/data/DATASET/'
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()
    
    def load_dataset(self):
        img_paths = []
        gt_paths = []
        labels = [] # 0:good, 1: black, 2: wrinkle
        
        for label , i in ['good' , 'black', 'wrinkle']:
            path = (glob.glob(self.root + label + "/*.jpg"))
            label = 

def Train_Dataloader(dataset_select):
    #data_transforms : resize images and perform pixel normalization
    data_transforms = Transform(args.resize, args.imagesize)
    gt_transforms = GT_Transform(args.resize, args.imagesize)

    if dataset_select == 'MVTec':
        image_datasets = MVTecDataset(args, root=os.path.join(args.dataset_path), transform=data_transforms, gt_transform=gt_transforms, phase='train')
    elif dataset_select =='Private':
        image_datasets = BTADDataset(args, root=os.path.join(args.dataset_path,args.category), transform=data_transforms, gt_transform=gt_transforms, phase='train')
    train_loader = DataLoader(image_datasets, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    return train_loader

def Test_Dataloader(args):
    data_transforms = Transform(args.resize, args.imagesize)
    gt_transforms = GT_Transform(args.resize, args.imagesize)

    if args.dataset_category == 'MVTec':
        test_datasets = MVTecDataset(args, root=os.path.join(args.dataset_path), transform=data_transforms, gt_transform=gt_transforms, phase='test')
    elif args.dataset_category == 'BTAD':
        test_datasets = BTADDataset(args, root=os.path.join(args.dataset_path,args.category), transform=data_transforms, gt_transform=gt_transforms, phase='test')
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return test_loader