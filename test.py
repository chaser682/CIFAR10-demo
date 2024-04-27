import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from cnn_model import CNN

input = torch.randint(0, 256, size=(64, 3, 32, 32), dtype=torch.float32)
# input = input.numpy()
# torch.from_numpy(input)
transform = transforms.Compose([
    # transforms.Resize((32, 32)),
    # transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
input = transform(input)
model = CNN()
output = model(input)
print(input, input.size())
print(output, output.size())
predicted = output.max(1)[1] # output.max(1) 返回第一维度的最大值[0]和最大值下标[1]，为元组数据
print(predicted)

write = SummaryWriter('./logs')
write.add_graph(model, input)
write.close()