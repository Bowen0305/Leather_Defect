import pytorch_lightning as pl
import os

from utils.data.load_data import Train_Dataloader, Test_Dataloader, Distribution_Train_Dataloader, Coor_Distribution_Train_Dataloader
from utils.learning.train_part import Coreset, Distribution, AC_Model, Coor_Distribution
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(22)
train_dataloader, test_dataloader = Train_Dataloader(), Test_Dataloader()