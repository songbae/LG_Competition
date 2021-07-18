from albumentations.augmentations.crops.transforms import RandomCrop
from albumentations.augmentations.functional import cutout
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.transforms import HorizontalFlip, VerticalFlip
from albumentations.core.composition import OneOf
import numpy as np
import torch
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import random
import os
import yaml
import collections
from easydict import EasyDict


def tf_form(mode=None):
    trn_tfms = A.Compose(
        [
          # A.OneOf([
          #   A.HorizontalFlip(),
          #   A.VerticalFlip(),
          #   A.RandomRotate90()
          # ]),
          A.Cutout(p=0.2),
          A.Normalize(),
            ToTensorV2()
            ], additional_targets={'label': 'image'})
    val_tfms = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ], additional_targets={'label': 'image'}
    )
    if mode == 'train':
        return trn_tfms
    else:
        return val_tfms


def denormalize(img,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
  c,h,w= img.shape
  new_img=np.zeros_like(img)
  new_img=(img*std+mean)*255.0
  new_img=torch.from_numpy(new_img).permute(1,2,0)
  new_img=new_img.numpy()
  return  new_img

def denormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img_cp = image.copy()
    for i in range(3):
        img_cp[i] = (img_cp[i]*std[i]+mean[i])*255.0
    img_cp = torch.from_numpy(img_cp).permute(1, 2, 0).numpy()
    
    img_cp = np.clip(img_cp, 0, 255).astype(np.float16)
    return img_cp


def denormalize_hi(image,mask):
    img_cp=(image*255.)/mask
    img_cp = torch.from_numpy(img_cp).permute(1, 2, 0).numpy()

    img_cp = np.clip(img_cp, 0, 255).astype(np.float32)
    return img_cp


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class YamlConfigManager:
    def __init__(self, config_file_path='./configs/config.yaml'):
        super().__init__()
        self.values = EasyDict()
        if config_file_path:
            self.config_file_path = config_file_path
            self.reload()

    def reload(self):
        self.clear()
        if self.config_file_path:
            with open(self.config_file_path, 'r') as f:
                self.values.update(yaml.safe_load(f))

    def clear(self):
        self.values.clear()

    def update(self, yml_dict):
        for (k1, v1) in yml_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1

    def export(self, save_file_path):
        if save_file_path:
            with open(save_file_path, 'w') as f:
                yaml.dump(dict(self.values), f)


class Flags:
    """ Flags object.
    """

    def __init__(self, config_file):
        try:
            with open(config_file, 'r') as f:
                d = yaml.safe_load(f)
        except:
            d = config_file

        self.flags = dict_to_namedtuple(d)

    def get(self):
        return self.flags


def dict_to_namedtuple(d):
    """ Convert dictionary to named tuple.
    """
    FLAGSTuple = collections.namedtuple('FLAGS', sorted(d.keys()))

    for k, v in d.items():

        if k == 'prefix':
            v = os.path.join('./', v)

        if type(v) is dict:
            d[k] = dict_to_namedtuple(v)

        elif type(v) is str:
            try:
                d[k] = eval(v)
            except:
                d[k] = v

    nt = FLAGSTuple(**d)

    return nt


def save_model(model, version, type='loss'):
    save_path = os.path.join(f'./ckpts/{version}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_dir = os.path.join(save_path, f'best_{type}.pth')
    torch.save(model.state_dict(), save_dir)
