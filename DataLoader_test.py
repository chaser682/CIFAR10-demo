from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = datasets.CIFAR10(root='./dataset', train=False, transform=transforms.ToTensor(), download=False)

test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False, num_workers=0, drop_last=False)

image, label = test_data[0]
print(image.shape, label)

writer = SummaryWriter('logs')
for i, data in enumerate(test_loader):
    images, labels = data
    print(images.shape, labels)
    writer.add_images('test_images', images, i)
    break
writer.close()
