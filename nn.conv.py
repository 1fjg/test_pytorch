import torch
import cv2 as cv
import  torch.nn.functional as F
import numpy as np
img=cv.imread('D:/pic/myw.png',0)
h,w=img.shape[:2]
input=torch.tensor(img,dtype=torch.long)
# input=torch.tensor([[1,2,0,3,1],
#                     [0,1,2,3,1],
#                     [1,2,1,0,0],
#                     [5,2,3,1,1],
#                     [2,1,0,1,1]])

kernel=torch.tensor([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])
input=torch.reshape(input,(1,1,h,w))
kernel=torch.reshape(kernel,(1,1,3,3))
output=F.conv2d(input,kernel,padding=1)
output=output.squeeze().detach().numpy()
min=output.min()
max=output.max()
output=((output-min)/(max-min)*255).astype(np.uint8)
cv.imshow('input',img)
cv.imshow('output',output)
cv.waitKey()