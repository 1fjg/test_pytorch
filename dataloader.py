import torch.utils.data
import torch.utils.tensorboard
import torchvision

test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

test_loader= torch.utils.data.DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

img,target=test_data[0]
print(img.shape,target)

writer= torch.utils.tensorboard.SummaryWriter("dataloader")
step=0
for data in test_loader:
    img,target=data
    writer.add_images("data_loader_drop",img,step)
    step+=1
    # print(img.shape,target)

writer.close()