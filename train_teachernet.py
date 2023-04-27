import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
# 定义数据集

train_transforms=transforms.Compose([
                             transforms.Resize((256, 256)),
                             transforms.CenterCrop((256, 256)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#train_dataset = ImageFolder(root='/media/densogup-1/8T/jyp/myad/data/ILSVRC2012_img_train', transform=train_transforms)
#train_dataset = datasets.ImageNet(root='./data', train=True,  transform=train_transforms)
train_dataset = ImageFolder(root='/media/densogup-1/8T/jyp/myad/data/train', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
# 定义 teacher 网络
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 384, kernel_size=4, stride=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(384,19)
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.pool3(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        #print(x.size())
        return x



# 定义优化器和损失函数
lr = 0.0001
num_epochs = 10
teacher_net = TeacherNet()
teacher_net = teacher_net.cuda()
optimizer = optim.Adam(teacher_net.parameters(), lr=lr,weight_decay=0.00001)
criterion = nn.CrossEntropyLoss()
#criterion = nn.functional.mse_loss()

# 数据集上训练 teacher 网络
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = teacher_net(images)
        #labels = labels.view(-1,1)
        #labels = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        #print(outputs.size(), labels.size())
        loss = criterion(outputs.view(-1,19),labels.view(-1))
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# 保存预训练的 teacher 网络权重为 pth 文件
torch.save(teacher_net.state_dict(), 'teacher_net.pth')