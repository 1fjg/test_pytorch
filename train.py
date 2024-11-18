import torch.utils.data
import torch.utils.tensorboard
import torchvision
from model import Tudui#b再次修改
# b修改
#1修改
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data=torchvision.datasets.CIFAR10("data",train=True,download=True,
                                        transform=torchvision.transforms.ToTensor())

test_data=torchvision.datasets.CIFAR10("data",train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())

train_data_size=len(train_data)
test_data_size=len(test_data)

# print(f"训练集的长度为：{train_data_size}")
# print(f"测试集的长度为：{test_data_size}")

train_dataloader= torch.utils.data.DataLoader(train_data,batch_size=64)
test_dataloader=torch.utils.data.DataLoader(test_data,batch_size=64)


tudui=Tudui().to(device)
loss_fn=torch.nn.CrossEntropyLoss().to(device)

learning_rate=0.03
optimizer=torch.optim.SGD(tudui.parameters(),lr=learning_rate)

total_train_step=0
total_test_step=0
epcoh=10
writer= torch.utils.tensorboard.SummaryWriter("logs_train")
for i in range(epcoh):
    print(f"-----第{i+1}轮训练开始-----")
    loss_total=0
    for data in train_dataloader:
        imgs,targets=data
        imgs,targets=imgs.to(device),targets.to(device)
        outputs=tudui(imgs)
        loss=loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step+=1
        loss_total+=loss
        if total_test_step%100==0:
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    print(f"第{i+1}轮训练总损失：{loss_total}")

    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            imgs,targets=imgs.to(device),targets.to(device)
            outputs=tudui(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss+=loss
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy+=accuracy
        print(f"第{i+1}轮测试总损失： {total_test_loss}")
        print(f"第{i+1}轮测试正确率： {total_accuracy/test_data_size}")
        writer.add_scalar("test_loss",total_test_loss,total_test_step)
        writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
        total_test_step+=1

writer.close()