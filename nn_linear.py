import torch.nn
import torch.utils.tensorboard
import torchvision
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,download=True,
                                     transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset,batch_size=64,drop_last=True)

class Tudui(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1= torch.nn.Linear(196608,10)

    def forward(self,input):
        output=self.linear1(input)
        return output
# writer= torch.utils.tensorboard.SummaryWriter()
tudui=Tudui()
for data in dataloader:
    imgs,targets=data
    print(imgs.shape)
    input=torch.reshape(imgs,(1,1,1,-1))
    # input=torch.flatten(imgs)
    print(input.shape)
    output=tudui(input)
    print(output.shape)