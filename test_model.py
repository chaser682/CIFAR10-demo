import torch
from cnn_model import CNN

mode = CNN()
mode.load_state_dict(torch.load('./params/cnn_model_001_100.pth'))
mode.eval()
with torch.no_grad():
    data = torch.randn(1, 3, 32, 32)
    output = mode(data)
    print(output, output.size())