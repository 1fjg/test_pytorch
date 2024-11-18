import torch
import torchvision

vgg16=torchvision.models.vgg16(pretrained=False)

torch.save(vgg16,"vgg16_method1.pth")

torch.save(vgg16.state_dict(),"vgg16_method2.pth")

class Tudui(torch.nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.conv1=torch.nn.Conv2d(3,64,kernel_size=3)
    def forward(self,x):
        x=self.conv1(x)
        return x

tudui=Tudui()
torch.save(tudui,"tudui_method1.pth")