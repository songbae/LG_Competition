import albumentations as A
import cv2
from tqdm import tqdm
from glob import glob
import os
import numpy as np
train_lmg_list = sorted(glob('./data/train_input_img/*.png'))
train_label_list = sorted(glob('./data/train_label_img/*.png'))
cnt = 1
if not os.path.exists('./data/new_train_img2/'):
    os.mkdir('./data/new_train_img2/')
if not os.path.exists('./data/new_label_img2/'):
    os.mkdir('./data/new_label_img2/')
if not os.path.exists('./data/new_resize_img2/'):
    os.mkdir('./data/new_resize_img2/')
stride = 128
size = 384
for idx, (train, labels) in tqdm(enumerate(zip(train_lmg_list, train_label_list))):
    img = cv2.imread(train)
    label = cv2.imread(labels)
    h, w, c = img.shape
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            cnt += 1
            now_h = i
            now_w = j
            if now_h+size > h:
                now_h -= size
            if now_w+size > w:
                now_w -= size
            save_resize_img = np.zeros((size*3, size*3, 3))
            boarder = -np.ones((h+size*2, w+size*2, 3))
            boarder[size:size+h, size:size+w, :] = img
            save_img = img[now_h:now_h+size, now_w:now_w+size, :]
            save_label = label[now_h:now_h+size, now_w:now_w+size, :]
            H, W = now_h+size, now_w+size
            save_resize_img = boarder[H-size:H+size*2, W-size:W+size*2, :]
            save_resize_img = cv2.resize(save_resize_img, dsize=(
                size, size), interpolation=cv2.INTER_LINEAR)

            if np.mean(save_img) < 8:
                continue

            cv2.imwrite(f'./data/new_train_img2/{str(cnt)}_r.png', save_img)
            cv2.imwrite(f'./data/new_label_img2/{str(cnt)}_r.png', save_label)
            cv2.imwrite(
                f'./data/new_resize_img2/{str(cnt)}_r.png', save_resize_img)
