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
from torch.utils.data import DataLoader, Dataset
import cv2
from src.dataset import *
from src.losses import *
from src.scheduler import *
from src.utils import *
from model.network import *
from model.Hinet import *
from collections import OrderedDict


def main(config):
  options = Flags(config).get()
  # options=YamlConfigManager(config)

  # set wandb
  wandb.run.name = options.swin.name
  wandb.run.save()
  # wandb.config.update(options.swin)

  # fix seed
  seed_everything(options.swin.seed)
  # use gpu
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  train_tfms = tf_form(mode='train')
  valid_tfms = tf_form(mode='valid')
  tr_loader, valid_loader, tr_dataset, valid_dataset = dataset_loader(
      options.swin, train_tfms, valid_tfms)
  ### train &valid


  model = HINet()
  optimizer = optim.AdamW(model.parameters(), lr=options.swin.lr,weight_decay=0.000001)
  scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=100,
                                            max_lr=options.swin.train.max_lr, min_lr=options.swin.train.min_lr, gamma=options.swin.train.gamma)
  wandb.watch(model)
  best_score = 0.0
  best_loss = float('inf')
  criterion = PSNRLoss(loss_weight=0.5,reduction='mean').to(device)
  model.to(device)

  for epoch in range(options.swin.train.num_epochs):
    for phase in ['train', 'valid']:
      run_loss = []
      psnr_loss = 0.0
      if phase == 'train':
        model.train()
        now_dl = tr_loader
      else:
        model.eval()
        now_dl = valid_loader
      with torch.set_grad_enabled(phase == 'train'):
        with tqdm(now_dl, total=len(now_dl), unit='batch') as now_bar:
          for batch, sample in enumerate(now_bar):
            now_bar.set_description(f'{phase} Epoch {epoch}')
            optimizer.zero_grad()
            image, label = sample['image'].to(
                device), sample['label'].to(device).type(torch.float32)
            preds = model(image)
            if not isinstance(preds,list):
              preds=[preds]
            output=preds[-1]
            loss=0
            loss_dict=OrderedDict()
            # pixel_loss
            l_pix=0.
            for pred in preds:
              l_pix+=criterion(pred,label)
            loss+=l_pix
            loss_dict['l_pix']=l_pix
            loss=loss+0*sum(p.sum() for p in model.parameters())
            if phase == 'train':
              loss.backward()
              nn.utils.clip_grad_norm_(model.parameters(),0.01)
              optimizer.step()
            batch_loss =run_loss
            now_bar.set_postfix(psnr_loss=batch_loss)
            run_loss.append(batch_loss)
          epoch_loss=np.mean(run_loss)

          if phase == 'valid':
            scheduler.step()
            wandb.log({
                'valid_loss': epoch_loss,
            })
          if phase == 'train':
            wandb.log({
                'train_loss': epoch_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
          if phase == 'valid' and epoch_loss< best_loss:
            best_loss = epoch_loss
            save_model(model, options.swin.name)
            print('best_model saved')


if __name__ == '__main__':
  wandb.init(project='LG_camera_light', reinit=True)
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--config_file',
      dest='config_file',
      default='./configs/config.yaml',
      type=str,
      help='Path of config file'
  )
  parser = parser.parse_args()
  main(parser.config_file)
