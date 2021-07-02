# -*- coding: utf-8 -*-
"""
Created on Tue May 18 19:39:58 2021

@author: part
"""

from PIL import Image
import matplotlib.pyplot as plt 
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import resnet50

img = Image.open('pig.jpg')
T = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
transformed_img = T(img)[None,:,:,:]
plt.imshow(transformed_img[0].numpy().transpose(1,2,0))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
input_img = normalize(transformed_img)
model = resnet50(pretrained=True)
model.eval();
pred = model(input_img)

a = pred.detach().numpy()
max_index = a.argmax()
# print(max_index)


import json
with open('imagenet_class_index.json') as f:
    imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}

print(imagenet_classes[max_index])

print(nn.CrossEntropyLoss()(model(input_img),torch.LongTensor([341])).item())
##########################################################
# for input, target in dataset:
#     optimizer.zero_grad()
#     output = model(input)
#     loss = loss_fn(output, target)
#     loss.backward()
#     optimizer.step()
###########################################################

import torch.optim as optim

delta = torch.zeros_like(transformed_img,requires_grad=True)
eps = 2./255
opt = optim.SGD([delta], lr=0.1)

for i in range(30):
    pred = model(normalize(transformed_img+delta))
    Loss = -nn.CrossEntropyLoss()(pred,torch.LongTensor([341]))
    
    opt.zero_grad()
    Loss.backward()
    opt.step()
    delta.data.clamp_(-eps, eps)
    if i%5 ==0:
        print(i,Loss.item())
    
print("True class probability:", nn.Softmax(dim=1)(pred)[0,341].item())
    
plt.imshow((transformed_img+delta)[0].detach().numpy().transpose(1,2,0))
a2 = pred.detach().numpy()
max_index = a2.argmax()
print(imagenet_classes[max_index])
plt.imshow((50*(delta)[0]+0.5).detach().numpy().transpose(1,2,0))
#########################################################################
delta1 = torch.zeros_like(transformed_img,requires_grad=True)
eps = 2./255
opt = optim.SGD([delta1], lr=5e-3)

for i in range(100):
    pred = model(normalize(transformed_img+delta1))
    Loss = (-nn.CrossEntropyLoss()(pred,torch.LongTensor([341]))+
            nn.CrossEntropyLoss()(pred,torch.LongTensor([1])))
    
    opt.zero_grad()
    Loss.backward()
    opt.step()
    delta.data.clamp_(-eps, eps)
    if i%10 ==0:
        print(i,Loss.item())
    
print("True class probability:", nn.Softmax(dim=1)(pred)[0,341].item())
    
plt.imshow((transformed_img+delta1)[0].detach().numpy().transpose(1,2,0))
a3 = pred.detach().numpy()
max_index = a3.argmax()
print(imagenet_classes[max_index])
