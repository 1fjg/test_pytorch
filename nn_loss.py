import torch

inputs=torch.tensor([1,2,3],dtype=torch.float32)
targets=torch.tensor([1,2,5],dtype=torch.float32)

inputs=torch.reshape(inputs,(1,1,1,3))
targets=torch.reshape(targets,(1,1,1,3))

loss=torch.nn.MSELoss()
result=loss(inputs,targets)
print(result)

x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([0])
x=torch.reshape(x,(1,3))
loss_cross=torch.nn.CrossEntropyLoss()
result_cross=loss_cross(x,y)
print(result_cross)