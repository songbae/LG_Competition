from albumentations.augmentations import transforms
from pandas.core.frame import DataFrame
import torch 
from torch.utils.data import DataLoader,Dataset
import numpy as np 
import pandas as pd 
import cv2
import os 
from tqdm import tqdm 
from glob import glob 
from sklearn.model_selection import train_test_split
import random
import copy
class MyDataset(Dataset):
  """Some Information about MyDataset"""
  def __init__(self,options,csv, tfms=None):
    super(MyDataset, self).__init__()
    self.input=csv
    self.image=self.input[:,0]
    self.label=self.input[:,1]
    self.path=options.path
    self.label_path=options.label_path
    self.tfms=tfms
  def __len__(self):
    return len(self.input)
  
  def __getitem__(self,idx):
    img=cv2.imread(self.image[idx])
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    label=cv2.imread(self.label[idx])
    label=cv2.cvtColor(label,cv2.COLOR_BGR2RGB)
    image=img.astype(np.float32)
    label=label.astype(np.float32)
    name=self.image[idx]
    if self.tfms:
      transformed=self.tfms(image=image,label=label)
      image=transformed['image']
      label=transformed['label']
    sample={
      'image':image,
      'label':label,
      'name': name,
      # 'base':img
    }
    return sample


def np_rotate(img, label):
  k = np.random.randint(4)
  inp = np.rot90(img, k)
  la = np.rot90(label, k)
  return inp, la


def np_flip(img, label):
  f = np.random.randint(2)
  if f == 0:
    img = np.fliplr(img)
    label = np.fliplr(label)
  return img, label


def np_cut(img, label):
  h, w, c = img.shape
  new_h = np.random.randint(h-256)
  new_w = np.random.randint(w-256)
  inp = img[new_h:new_h+256, new_w:new_w+256, :]
  la = label[new_h:new_h+256, new_w:new_w+256, :]
  return inp, la

from glob import glob
import pandas as pd
def augmentation(img, label):
  inp, la = np_cut(img, label)
  inp, la = np_rotate(inp, la)
  inp, la = np_flip(inp, la)
  return inp, la
def dataset_loader(options, train_tfms, valid_tfms,num_wokers=4):
  train_path_list=glob('./data/new_train_img/*.png')
  label_path_list = glob('./data/new_label_img/*.png')
  train_csv=pd.DataFrame(train_path_list)
  label_csv=pd.DataFrame(label_path_list)
  csv=pd.concat([train_csv,label_csv],axis=1,ignore_index=True)
  train,valid=train_test_split(csv.values,test_size=0.1)
  train_dataset=MyDataset(options,train,tfms=train_tfms)
  valid_dataset=MyDataset(options,valid,tfms=valid_tfms)
  train_data_loader=DataLoader(train_dataset,batch_size=options.train.batch_size,shuffle=True,num_workers=num_wokers,drop_last=True)
  valid_data_loader=DataLoader(valid_dataset,batch_size=options.train.batch_size,shuffle=False,num_workers=num_wokers,drop_last=True)
  return train_data_loader, valid_data_loader, train_dataset,valid_dataset
