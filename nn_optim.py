import torch.nn
import torch
import torch.utils.data
import torchvision

dataset=torchvision.datasets.CIFAR10('dataset',train=False,download=True,
                                     transform=torchvision.transforms.ToTensor())
dataloader= torch.utils.data.DataLoader(dataset,batch_size=1)
class Tudui(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model1= torch.nn.Sequential(
            torch.nn.Conv2d(3,32,5,padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5,padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1024,64),
            torch.nn.Linear(64,10)
        )
    def forward(self,x):
        x=self.model1(x)
        return x

import torch




tudui=Tudui()
device=torch.device('cuda:0')
tudui=tudui.to(device)
loss=torch.nn.CrossEntropyLoss()
optim=torch.optim.SGD(tudui.parameters(),lr=0.01)
for epoch in range(20):
    running_loss=0.0
    for i,data in enumerate(dataloader):
        if(i!=2):continue
        imgs,targets=data
        imgs, targets = imgs.to(device), targets.to(device)
        outputs=tudui(imgs)
        result_loss=loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss+=result_loss
    print(running_loss)

for i,data in enumerate(dataloader):
    if(i!=2):continue
    imgs,targets=data
    imgs, targets = imgs.to(device), targets.to(device)
    outputs=tudui(imgs)
    result_loss=loss(outputs,targets)
    print(outputs)
    print(targets)

