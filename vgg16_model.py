from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch

vgg16 = models.vgg16(pretrained=True, progress=True)
vgg16.classifier.add_module('add_linear', nn.Linear(1000, 10))
vgg16.classifier.add_module('softmax', nn.LogSoftmax(dim=1))
print(vgg16)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= 64, shuffle=True, num_workers=0, drop_last=True)
model = vgg16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

for epoch in range(100):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        # 将当前步骤网络模型参数梯度清零
        optimizer.zero_grad()
        # 梯度下降
        loss.backward()
        # 进行下一步
        optimizer.step()
        if i % 10 == 0:
            _, predicted = torch.max(outputs.data, 1)
            equals = predicted == labels.data
            accuracy = torch.mean(equals * 1.0)
            print('[%d, %5d] loss: %5f, accuracy: %5f' % (epoch + 1, i + 1, loss.item(), accuracy.item()))
            # print(outputs)
            # print(labels)
            # print(predicted)