import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils


def train(model, train_loader, valid_loader, num_epochs, out_folder):

  # Create new folder to save all the weights
  new_folder_path = os.path.join("/content", out_folder)
  os.mkdir(new_folder_path)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.to(device)

  # Which params to freeze
  for param in model.backbone.parameters():
    param.requires_grad = False


  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)

  for epoch in range(num_epochs):

      train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
      evaluate(model, valid_loader, device=device)

      if ((epoch + 1) % 10 == 0) or ((epoch + 1) == num_epochs):
        filename = "checkpoint_{}.pth".format(epoch + 1)
        SAVE_PATH = os.path.join(new_folder_path, filename)
        save_model(model, epoch, SAVE_PATH)
