# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:44:09 2021

@author: part
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from torchvision import transforms

pig_img = Image.open("pig.jpg")
preprocess = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
pig_tensor = preprocess(pig_img)
# plot image (note that numpy using HWC whereas Pytorch user CHW, so we need to convert)
plt.imshow(pig_tensor.numpy().transpose(1,2,0))

import torch
import torch.nn as nn
from torchvision.models import resnet50

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

# values are standard normalization for ImageNet images, 
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# # load pre-trained ResNet50, and put into evaluation mode (necessary to e.g. turn off batchnorm)
model = resnet50(pretrained=True)
model.eval();

# # form predictions
pred = model(norm(pig_tensor))
import json 
with open("imagenet_class_index.json") as f:
 imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}

print(imagenet_classes[pred.max(dim=1)[1].item()])

# print(nn.CrossEntropyLoss()(model(norm(pig_tensor)),torch.LongTensor([341])).item())