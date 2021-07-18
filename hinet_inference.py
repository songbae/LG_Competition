import os
from model.network import *
from model.Unext50 import UneXt50
from model.Hinet import *
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import cv2
from tqdm import tqdm
import zipfile
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from src.utils import denormalize_hi
import gc
import PIL


class tesedataset(Dataset):
    def __init__(self, img_path, tfms, size=256):
        super().__init__()
        self.img_path = img_path
        self.tfms = tfms
        self.size = size

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, x):
        img = cv2.imread(self.img_path[x])
        img = img.astype(np.float32)/255.
        h, w, c = img.shape
        crop_list = []
        add_list=[]
        stride = 32
        size = self.size
        for i in range(0, h, stride):
            temp_crop_img = []
            add_resize_img=[]
            for j in range(0, w, stride):
                new_h = i
                new_w = j
                if new_h+size > h:
                    new_h = h-size
                if new_w+size > w:
                    new_w = w-size
                # assert new_w==224 and new_h==224,print(new_w,new_h)
                save_resize_img=np.zeros((256*3,256*3,3))
                boarder=np.zeros((h+512,w+512,3))
                boarder[256:256+h,256:256+w,:]=img
                H, W = new_h+256, new_w+256
                save_resize_img=boarder[H-256:H+512,W-256:W+512,:]
                save_resize_img=cv2.resize(save_resize_img,dsize=(256,256),interpolation=cv2.INTER_LINEAR).astype(np.float32)
                crop_img = img[new_h:new_h+size, new_w:new_w+size, :]
                crop_img = self.tfms(image=crop_img)[
                    'image']  # 이렇게 하면 텐서까지 변환 완료
                save_resize_img=self.tfms(image=save_resize_img)['image']
                temp_crop_img.append(crop_img)
                add_resize_img.append(save_resize_img)
            crop_list.append(temp_crop_img)
            add_list.append(add_resize_img)
        sample = {
            'image': crop_list,
            'add_img':add_list,
            'name': f'test_{20000+x}.png'
        }
        return sample


def tf_form():
    test_tfms = A.Compose([ToTensorV2()])
    return test_tfms


test_csv = pd.read_csv('./data/test.csv')
test_input_files = './data/test_input_img/'+test_csv['input_img']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HINet()
model.load_state_dict(torch.load('./ckpts/HINET/best_8.pth'))
model.eval().to(device)
test_tfms = tf_form()
test_dt = tesedataset(test_input_files, test_tfms)
train_data_loader = DataLoader(
    test_dt, batch_size=1, shuffle=False, num_workers=0)
submission_path = './ckpts/submission_hinet_8/'
if not os.path.exists(submission_path):
    os.mkdir(submission_path)

# inference code
for idx, sample in enumerate(tqdm(train_data_loader)):
    if idx == 0 or idx == 1 or idx == 2 or idx == 7:continue
    img_list, name = sample['image'], sample['name'][0]
    add_list=sample['add_img']
    result_img = np.zeros((3, 2448, 3264), dtype=np.float32)
    mask_img = np.zeros((3, 2448, 3264), dtype=np.float32)
    size = 256
    stride = 32
    for idx, i in enumerate(img_list):
        for jdx, img in enumerate(i):
            h = idx*stride if idx*stride+size <= 2448 else 2448-size
            w = jdx*stride if jdx*stride+size <= 3264 else 3264-size
            preds = model(img.to(device),add_list[idx][jdx].to(device))
            preds = preds[-1].detach().cpu().numpy()
            result_img[:, h:h+size, w:w+size] += preds[0]
            mask_img[:, h:h+size, w:w+size] += 1
    result_img = denormalize_hi(result_img, mask_img).astype(np.uint8)
    cv2.imwrite(os.path.join(submission_path, name), result_img)
