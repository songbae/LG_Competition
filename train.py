from model.Unext50 import UneXt50
from model.hrnet import Hrnet
import os 
import wandb 
import argparse
import torch 
import torch.optim as optim 
import torch.nn.functional as F
import yaml 
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
import cv2 
from src.dataset import *
from src.losses import *
from src.scheduler import *
from src.utils import *
from model.network import * 
from collections import OrderedDict

def main(config):
  options=Flags(config).get()
  # options=YamlConfigManager(config)

  # set wandb
  wandb.run.name=options.swin.name
  wandb.run.save()
  # wandb.config.update(options.swin)

  # fix seed 
  seed_everything(options.swin.seed)
  # use gpu
  device= 'cuda' if torch.cuda.is_available() else 'cpu'
  train_tfms=tf_form(mode='train')
  valid_tfms=tf_form(mode='valid')
  tr_loader,valid_loader,tr_dataset,valid_dataset=dataset_loader(options.swin, train_tfms,valid_tfms)
  ### train &valid

  # model=Unet()
  # model=SwinUnet()
  # model.load_from('./model/pretrained/swin_tiny_patch4_window7_224.pth')
  # with open('./configs/hrnet_config.yaml') as f:
  #   cfg=yaml.load(f)
  # model=Hrnet(cfg)
  model=UneXt50()
  model.load_state_dict(torch.load('./ckpts/Unext/best_1.pth'))
  optimizer=optim.AdamW(model.parameters(),lr=options.swin.lr,weight_decay=0.000001)
  scheduler=CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=40000,max_lr=options.swin.train.max_lr,min_lr=options.swin.train.min_lr,gamma=options.swin.train.gamma)
  wandb.watch(model)
  best_score=0.0
  best_loss=float('inf')
  criterion=nn.L1Loss()
  model.to(device)
  
  for epoch in range(options.swin.train.num_epochs):
    for phase in ['train','valid']:
      run_loss=0.0
      psnr_loss=list()
      if phase=='train':
        model.train()
        now_dl=tr_loader
      else:
        model.eval()
        now_dl=valid_loader
      with torch.set_grad_enabled(phase=='train'):
        with tqdm(now_dl ,total=len(now_dl),unit='batch') as now_bar:
          for batch, sample in enumerate(now_bar):
            now_bar.set_description(f'{phase} Epoch {epoch}')
            optimizer.zero_grad()
            image,label=sample['image'].to(device),sample['label'].to(device).type(torch.float32)
            preds=model(image)
            if isinstance(preds,OrderedDict):
              loss=criterion(preds,label)
            elif isinstance(preds,list):
              for i in range(len(preds)):
                pred=preds[i]
                ph,pw=pred.size(2),pred.size(3)
                h,w=label.size(2), label.size(3)
                if ph != h or pw != w:
                    pred = F.interpolate(input=pred, size=(
                        h, w), mode='bilinear', align_corners=True)
                preds[i]=pred
            if isinstance(preds,list):
              loss=0.0
              for i in range(len(preds)):
                loss+=criterion(preds[i],label)*(1/2**i)
              preds=pred[0]
            else:
              loss=criterion(preds,label)
            if phase=='train':
              loss.backward()
              optimizer.step()
              scheduler.step()
            batch_loss=rmse_score(preds.detach().cpu().numpy(), label.detach().cpu().numpy())
            run_loss+=batch_loss
            psnr_loss.append(psnr_score(batch_loss))
            now_bar.set_postfix(loss=np.mean(psnr_loss))

              

          if phase=='valid':
            wandb.log({
              'valid_loss':run_loss,
              'valid_psnr':psnr_loss
            })
          if phase=='train':
            wandb.log({
              'train_loss':run_loss,
              'train_psnr':psnr_loss,
              'learning_rate':optimizer.param_groups[0]['lr']
            })
          if phase=='valid' :
            save_model(model,options.swin.name,type=epoch)
            print('best_model saved')



            

if __name__=='__main__':
  wandb.init(project='LG_camera_light', reinit=True)
  parser=argparse.ArgumentParser()
  parser.add_argument(
    '--config_file',
    dest='config_file',
    default='./configs/config.yaml',
    type=str, 
    help='Path of config file'
  )
  parser=parser.parse_args()
  main(parser.config_file)
