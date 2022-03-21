#网上找的版本可以跑的通
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

num_epochs = 5
batch_size = 100
learning_rate = 0.001

# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

train_dataset = datasets.MNIST(root='./mnist/',  # 数据集保存路径
                                      train=True,  # 是否作为训练集
                                      transform=transforms.ToTensor(),  # 数据如何处理, 可以自己自定义
                                      download=True)  # 路径下没有的话, 可以下载
 
test_dataset = datasets.MNIST(root='./mnist/',
                                     train=False,
                                     transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=2))
        
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,10))            

 
    def forward(self, x):
            x = self.conv1(x)
            x = x.view(-1,14*14*128)
            x = self.dense(x)
            return x    
        
        
cnn = Model()
if torch.cuda.is_available():
    cnn = cnn.cuda()
# 选择损失函数和优化方法
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = get_variable(images)
        labels = get_variable(labels)
        #这个地方其实感觉也可以写成inputs, labels = inputs.to(device), labels.to(device)
        #同时上面的if torch.cuda.is_available():
        #              cnn = cnn.cuda()应该可以改成cnn=cnn.to(device)这是我的猜想
 
        outputs = cnn(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
 
 
# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')