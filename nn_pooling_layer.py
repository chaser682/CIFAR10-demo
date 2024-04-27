import torch
from torch import nn

pool = nn.MaxPool2d(kernel_size=2, stride=2)
input = torch.randn(64, 3, 32, 32, dtype=torch.float32)
output = pool(input)
print(output, output.size())