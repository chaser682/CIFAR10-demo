import torch
from torch import nn

relu = nn.ReLU(inplace=False)
input = torch.tensor([[1.0, 2.0],
                      [-3.0, -4.0]], dtype=torch.float32)
output = relu(input)
print(output, output.size())

sigmoid = nn.Sigmoid()
output = sigmoid(input)
print(output, output.size())