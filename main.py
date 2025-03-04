import pytorch_lightning as pl
import argparse
import os

from utils.data.load_data import Train_Dataloader, Test_Dataloader, Distribution_Train_Dataloader, Coor_Distribution_Train_Dataloader
from utils.learning.train_part import Coreset, Distribution, AC_Model, Coor_Distribution
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    #vars
    seed = 22
    root_dir = None
    emb_dir = './embedding'


    pl.seed_everything(seed)
    train_dataloader, test_dataloader = Train_Dataloader, Test_Dataloader

    print("Start generating embedding coreset and distribution coreset")
    coreset_generator_trainer = pl.Trainer()
    coreset_generator = Coreset()
    coreset_generator_trainer.fit(coreset_generator , train_dataloaders = train_dataloader)
    print("End generating embedding coreset and distribution coreset")

