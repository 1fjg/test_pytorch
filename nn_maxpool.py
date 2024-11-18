import torch
import torch.nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision
input=torch.tensor([[1,2,0,3,1],
                         [0,1,2,3,1],
                         [1,2,1,0,0],
                         [5,2,3,1,1],
                         [2,1,0,1,1]],dtype=torch.float32)
dataset=torchvision.datasets.CIFAR10('./dataset',train=False,download=True,
                                     transform=torchvision.transforms.ToTensor())
dataloader= torch.utils.data.DataLoader(dataset,batch_size=64)
# input=torch.reshape(input,(1,5,5))

class Tudui(torch.nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.maxpool1= torch.nn.MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool1(input)
        return output

tudui=Tudui()

writer=torch.utils.tensorboard.SummaryWriter("logs_maxpool")
for i,data in enumerate(dataloader):
    imgs,targets=data
    writer.add_images("input",imgs,i)
    output=tudui(imgs)
    writer.add_images("output",output,i)

writer.close()
