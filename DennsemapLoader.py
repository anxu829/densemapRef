import gluoncv
from mxnet.gluon.data import *
from mxnet import autograd, gluon, image, init, nd
import glob
import numpy as np 
# import cv2
import pandas as pd 
from gluoncv.data.batchify import Tuple,Append
from mxnet.gluon import data as gdata

from mxnet.gluon.data import DataLoader

class DensemapDataset(Dataset):

    def __init__(self,csvPath = './gt_csv/*.csv' , imgPath = './img/*.jpg'):
        self._imgPath = imgPath
        self._csvPath = csvPath
        self._imglist = [file for file in glob.iglob(imgPath)]
        self._csvPath = [file for file in glob.iglob(csvPath)]
        self._len = len(self._imglist)

    def __getitem__(self,idx):
        img = image.imread(self._imglist[idx])
        label = pd.read_csv(self._csvPath[idx])
        return (img,label)


    def __len__(self):
        return self._len


if __name__ == "__main__":
    batch_size = 2 
    batchify_fn = Tuple(Append(), Append())
    train_dataset = DensemapDataset()
    im = train_dataset[0]

    def train_transform(*trans_data):
        img = trans_data[0]
        aug = gdata.vision.transforms.RandomFlipLeftRight()
        return (aug(img),trans_data[1])

    train_loader = DataLoader(train_dataset.transform(train_transform), batch_size = 2, shuffle=True, batchify_fn=batchify_fn)
    for ib, batch in enumerate(train_loader):
        print(type(batch[0][0]) ,type(batch[0][1]) )
    
    