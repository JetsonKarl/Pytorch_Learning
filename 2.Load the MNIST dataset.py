import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.repeat(3,1,1)),
                              transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
data_train=datasets.MNIST(root = "./data/",
                          transform=transform,
                          train=True,
                          download=True)
data_test=datasets.MNIST(root = "./data/",
                          transform=transform,
                          train=False)

data_loader_train=torch.utils.data.DataLoader(dataset=data_train,
                                              batch_size=64,
                                              shuffle=True)
data_loader_test=torch.utils.data.DataLoader(dataset=data_test,
                                             batch_size=64,
                                             shuffle=True)

images,labels=next(iter(data_loader_train))
img=torchvision.utils.make_grid(images)
img=img.numpy().transpose(1,2,0)
std=[0.5,0.5,0.5]
mean=[0.5,0.5,0.5]
img=img*std+mean
print([labels[i] for i in range(64)])
plt.imshow(img)