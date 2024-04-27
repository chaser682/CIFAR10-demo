import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 模型保存方式1：模型结构 + 模型参数
vgg16 = models.vgg16(pretrained=False)
print(vgg16)
torch.save(vgg16, './param/vgg16_model1.pth')

# 模型保存方式2：模型参数
torch.save(vgg16.state_dict(), './param/vgg16_model2.pth')


# 模型加载方式1：
model = torch.load('./param/vgg16_model1.pth')
print(model)

# 模型加载方式2：
model1 = models.vgg16(pretrained=False)
model1.load_state_dict(torch.load('./param/vgg16_model2.pth'))
# model1 = torch.load('./param/vgg16_model2.pth')
print(model1)

# acc = 0.99383
# print('Accuracy is: %5f%% ' % (100 * acc))