import torch
from torch.autograd import Variable
batch_n=100
hidden_layer=100
input_data=1000
output_data=10


#一批次要输入batch_n个数据 ，每个数据包含的数据特征有input_data个 
# hidden_layer用于定义经过隐藏层后薄酒的数据特征的个数 ，output_data是输出的数据，可以视为分类结果

#先输入100个具有1000个特征的数据， 经过隐藏层后变成100个具有100个特征的数据，再经过输出层后输出100个具有10个 分类结果值的数据
# 在得到输出结果之后计算损失并进行后向传播,这样一次模型的训练 就完成了，然后循环这个流程就可以完成指定次数的训练，并达到优化模型参数的目的

x = Variable(torch.randn(batch_n,input_data),requires_grad=False)
y = Variable(torch.randn(batch_n,output_data),requires_grad=False)
#以上代码中定义的从输入层到隐藏层、从隐藏层到输出层对应的权重参数，自动梯度需要用到torch.autograd包中的Variable类


#wl =Variable(torch.randn(input_data,hidden_layer),requires_grad=True)
#w2 =Variable(torch.randn(hidden_layer,output_data),requires_grad=True)
#因为有了torch.nn所以权重可以不写，由torch.nn自动生成

models = torch.nn.Sequential(
    torch.nn.Linear(input_data,hidden_layer),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer,output_data)
)

#（1）torch.nn.Sequential:是一种序列容器，嵌套神经网络中具体功能相关的类
#模块的加入一般有两种方式，一种是以上代码的直接嵌套，还有一种是以orderdict有序字典的方式进行传入，后者搭建的模型每一个模块都有我们自定义的名字

#（2）torch.nn.Linear:接收的参数有三个，分别是输入特征数、输出特征数、是否使用偏置
#第三个参数是一个布尔值，默认为True，将输入和输出特征数传递之后，自动生成比简单随机方式更好的参数初始化权重

#（3）torch.nn.ReLU：属于非线性激活分类，在定义时不需要传入参数

epoch_n=10000
learning_rate=1e-6
loss_fn=torch.nn.MSELoss()

#定义训练总次数和学习速率

optimzer=torch.optim.Adam(models.parameters(),lr=learning_rate)

for epoch in range (epoch_n):
    y_pred=models(x)
    loss=loss_fn(y_pred,y)
    if epoch%1000 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch,loss.data))
    optimzer.zero_grad()    
    
    loss.backward()
    optimzer.step()
    

#for param in models.parameters():
#    param.data-=param.grad.data*learning_rate
        
#w1.data-=learning_rate*w1.grad.data
#w2.data-=learning_rate*w2.grad.data    
#w1.grad.data.zero()
#w2.grad.data.zero()