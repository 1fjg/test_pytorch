import torch.utils.tensorboard
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True,transform=dataset_transform)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True,transform=dataset_transform)

writer= SummaryWriter("p10")
for i in range(10):
    img,target=test_set[i]
    print(test_set.classes[target])
    writer.add_image("test_img",img,i)

writer.close()
# print(test_set[0])
# img,label=test_set[0]
# print(train_set)
# print(test_set)
