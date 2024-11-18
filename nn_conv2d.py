import torch
import torch.nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                     download=True)
dataloader= torch.utils.data.DataLoader(dataset,batch_size=64)

class Tudui(torch.nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.conv1= torch.nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x

tudui=Tudui()
step=0
writer= torch.utils.tensorboard.SummaryWriter("./logs")
for data in dataloader:
    imgs,targets=data
    output=tudui(imgs)
    writer.add_images("input",imgs,step)
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step+=1