import cv2
import numpy as np
import os 
import pandas as pd 
from glob import glob
# sub_image = cv2.imread('./ckpts/submission/test_20003.png')
# sub_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB).astype(np.float32)
# base_image = cv2.imread('./data/test_input_img/test_input_20003.png')
# base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB).astype(np.float32)
# weight_image = (sub_image*4+base_image)/5.
# weight_image=cv2.cvtColor(weight_image,cv2.COLOR_RGB2BGR).astype(np.float32)
# sub_image = cv2.imread('weight_img3.png')
# sub_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB).astype(np.float32)
# print(sub_image)
# cv2.imwrite('./weight_img3.png',weight_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import albumentations as A 
import cv2 
from tqdm import tqdm
train_lmg_list=glob('./data/train_input_img/*.png')
train_label_list=glob('./data/train_label_img/*.png')
cnt=0
###이미지 사이즈
split_size=256
if not os.path.exists('./data/new_train_img/'):
  os.mkdir('./data/new_train_img/')
if not os.path.exists('./data/new_label_img/'):
  os.mkdir('./data/new_label_img/')
for idx,(train, labels) in tqdm(enumerate(zip(train_lmg_list,train_label_list))):
  img=cv2.imread(train)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  label=cv2.imread(labels)
  label=cv2.cvtColor(label,cv2.COLOR_BGR2RGB)
  stride=128
  size=split_size
  h,w,c=img.shape
  for i in range(0,h,stride):
    for j in range(0,w,stride):
      cnt+=1
      now_h=i
      now_w=j
      if now_h+size>h: now_h-=size
      if now_w+size>w: now_w-=size
      save_img=img[now_h:now_h+size,now_w:now_w+size,:]
      save_label =label[now_h:now_h+size, now_w:now_w+size, :]
      cv2.imwrite(f'./data/new_train_img/{str(cnt)}.png',save_img)
      cv2.imwrite(f'./data/new_label_img/{str(cnt)}.png', save_label)
  # resize_img=np.resize(img,(h//2,w//2,c))
  # resize_label =np.resize(label, (h//2, w//2, c))
  # h,w,c=resize_img.shape
  # stride=128
  # for i in range(0, h, stride):
  #   for j in range(0, w, stride):
  #     cnt += 1
  #     now_h = i
  #     now_w = j
  #     if now_h+size > h:
  #       now_h -= size
  #     if now_w+size > w:
  #       now_w -= size
  #     save_img = resize_img[now_h:now_h+size, now_w:now_w+size, :]
  #     save_label = resize_label[now_h:now_h+size, now_w:now_w+size, :]
  #     cv2.imwrite(f'./data/new_train_img/{str(cnt)}re.png',
  #                 save_img)
  #     cv2.imwrite(f'./data/new_label_img/{str(cnt)}re.png',
  #                 save_label)


