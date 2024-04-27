## 完整模型训练步骤

### 1.准备数据集

- 导入datasets包中的数据集
```python
import torchvision.datasets

train_data = torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=torchvision.transforms.ToTensor,download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor,download=True)
```

- 继承Dataset类的自定义数据集
```python
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.targets[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, target
```

### 2.加载数据集

```python
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, drop_last=False)
test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, drop_last=False)
```

### 3.搭建神经网络

```python
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO

    def forward(self, x):
        # TODO

        return x


model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### 4.损失函数

```python
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()
```

### 5.优化器

```python
import torch.optim as optim
LR = 0.001
optimizer = optim.SGD(model.parameters(), lr=LR)
```

### 6.设置训练网络的超参数

```python
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练次数
EPOCH = 100
```

### 7.模型训练与测试

```python
import torch
from torch.utils.tensorboard import SummaryWriter

write = SummaryWriter('./logs')
for epoch in range(EPOCH):
    print("---------第 {} 轮训练开始-------------".format(epoch + 1))

    # 训练步骤
    model.train()
    for i, data in enumerate(train_dataloader):
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 = 0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
            write.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_accuracy += (outputs.argmax(1) == targets).sum()
    total_test_step += 1
    print("整体测试集上的Loss：{}, 整体测试集上的Accuracy：{}%".format(total_test_loss, 100 * total_accuracy / len(test_data)))
    write.add_scalar("train_loss", total_test_loss, total_test_step)
    
    torch.save(model.state_dict(), "model_{}_{}.pth".format(LR, epoch))
    print("--------模型保存成功！----------")
write.close()
```