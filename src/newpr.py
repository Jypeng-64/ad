
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision

# 自定义网络模型
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 384, kernel_size=4, stride=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.conv4(x)
        return x

# 设置超参数
batch_size = 1
learning_rate = 0.001
num_epochs = 10

# 定义数据预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载ImageNet数据集
trainset = torchvision.datasets.ImageFolder(root='/media/densogup-1/8T/jyp/myad/data/ILSVRC2012_img_train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

# 构建模型
model = MyNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 开始预训练
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        labels = labels.expand_as(outputs)
        loss = criterion(outputs, labels.float())

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每100个batch输出一次损失
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# 输出权重文件
torch.save(model.state_dict(), 'mynet_weights.pth')