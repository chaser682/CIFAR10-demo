import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        output = x + 1
        return output


model = Model()
x = torch.tensor(1.0)
output = model(x)
print(output)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 0)
        # F.conv2d()
        """
        nn.Conv2d是卷积层对象的实例化，可以在训练过程中通过反向传播来更新参数。
        F.conv2d是卷积函数，只进行卷积操作并返回结果，不包含任何可学习的参数。
        """

    def forward(self, x):
        x = self.conv1(x)
        return x


model = Net()
input = torch.randn(1, 3, 5, 5)
output = model(input)
print(output, output.shape)


dataset = torchvision.datasets.CIFAR10('./dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


model = CNN()
write = SummaryWriter('./logs')
for i, data in enumerate(dataloader):
    images, labels = data
    outputs = model(images)
    print(images.shape)
    print(outputs.shape)
    write.add_images('input', images, i)
    write.add_images('output', outputs, i)
write.close()
