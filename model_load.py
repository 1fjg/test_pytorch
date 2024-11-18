import torch
import torchvision

# model=torch.load("vgg16_method1.pth")
# print(model)
vgg16=torchvision.models.vgg16(pretrained=False)
model=torch.load("vgg16_method2.pth")
vgg16.load_state_dict(model)
# print(vgg16)

import model_save
model=torch.load("tudui_method1.pth")
print(model)