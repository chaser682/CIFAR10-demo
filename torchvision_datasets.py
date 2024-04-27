import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

dataset_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5)),
])


train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=False)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform, download=False)

print(test_set.classes)
image, label = test_set[0]
print(image, label)
# image.show()
print(len(train_set), len(test_set))

writer = SummaryWriter('logs')
for i in range(10):
    img, label = test_set[i]
    writer.add_image('test', img, i)
    print(i)

writer.close()