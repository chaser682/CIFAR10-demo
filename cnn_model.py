import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from torch import optim


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convLayer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=2048, out_features=512, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=128, bias=True)
        self.fc4 = nn.Linear(in_features=128, out_features=10, bias=True)
        self.fcLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8192, out_features=8192, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(in_features=8192, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(in_features=1024, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(in_features=1024, out_features=10, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        x = self.convLayer3(x)
        # print(x.shape)
        x = self.fcLayer(x)
        return x


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

    LR = 0.001
    EPOCHS = 100
    PARAM_PATH = './params/cnn_model_001_100.pth'

    model = CNN()
    if os.path.exists(PARAM_PATH):
        checkpoint = torch.load(PARAM_PATH)
        model.load_state_dict(checkpoint)
        print("--------模型参数加载成功！--------")
    # model = models.vgg16(pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    writer = SummaryWriter(log_dir='./logs')

    start_time = time.time()
    total_train_epoch = 0
    total_test_epoch = 0
    for epoch in range(EPOCHS):
        # 开始训练
        model.train(True)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            # 将当前步骤网络模型参数梯度清零
            optimizer.zero_grad()
            # 梯度下降
            loss.backward()
            # 进行下一步
            optimizer.step()
            total_train_epoch += 1
            if (i + 1) % 100 == 0:
                end_time = time.time()
                print('[epoch: %d, i: %5d, total time spent: %5f], train loss: %5f' %
                      (epoch + 1, i + 1, end_time - start_time, loss.item()))
                writer.add_scalar('train_loss', loss.item(), total_train_epoch)

        # 开始测试
        total_accuracy = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_accuracy += (predicted == labels.data).sum().item()
                loss = loss_fn(outputs, labels)
                total_test_epoch += 1
                if (i + 1) % 100 == 0:
                    writer.add_scalar('test_loss', loss.item(), total_test_epoch)
            print('Accuracy of the network on the %d test images: %5f%%' %
                  (len(test_dataset), 100 * total_accuracy / len(test_dataset)))
            writer.add_scalar('test_accuracy', 100 * total_accuracy / len(test_dataset), epoch)

            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), PARAM_PATH)
                print("--------模型参数保存成功！--------")

    writer.close()