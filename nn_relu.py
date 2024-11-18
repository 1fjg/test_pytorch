import torch
import torch.nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision

input=torch.tensor([[1,-0.5],
                    [-1,3]])

input=torch.reshape(input,(-1,1,2,2))
dataset=torchvision.datasets.CIFAR10("./dataset",train=False,download=True,
                                     transform=torchvision.transforms.ToTensor())
dataloader= torch.utils.data.DataLoader(dataset,batch_size=64)
class Tudui(torch.nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.relu1= torch.nn.ReLU()
        self.sigmoid1= torch.nn.Sigmoid()
    def forward(self,input):
        output=self.sigmoid1(input)
        return output

tudui=Tudui()
writer= torch.utils.tensorboard.SummaryWriter("./logs_relu")
for i,data in enumerate(dataloader):
    imgs,targets=data
    writer.add_images("input",imgs,i)
    output=tudui(imgs)
    writer.add_images("output",output,i)
writer.close()
