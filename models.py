import torch.nn as nn


class PDN_S(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 384, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class PDN_M(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 384, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x

class Teacher(nn.Module):

    def __init__(self, size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if size == 'M':
                self.pdn = PDN_M()
        elif size == 'S':
                self.pdn = PDN_S()

    def forward(self, x):
        x = self.pdn(x)
        return x


class EncConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        x = self.enconv6(x)
        return x


class DecConv(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bilinear1 = nn.Upsample(scale_factor=3, mode='bilinear')
        self.bilinear2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bilinear3 = nn.Upsample(scale_factor=1.5, mode='bilinear')
        self.bilinear4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bilinear5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bilinear6 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bilinear7 = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.deconv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, 384, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.bilinear1(x.unsqueeze(1)).squeeze(1)  # nn.Upsample only 4D,we need append and then quit.
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = self.bilinear2(x)
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = self.bilinear3(x)
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = self.bilinear4(x)
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = self.bilinear5(x)
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = self.bilinear6(x)
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = self.bilinear7(x)
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = EncConv()
        self.decoder = DecConv()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Student(nn.Module):

    def __init__(self, size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ae = AutoEncoder()
        if size == 'M':
            self.pdn = PDN_M()
        elif size == 'S':
            self.pdn = PDN_S()

    def forward(self, x):
        pdn_out = self.pdn(x)
        ae_out = self.ae(x)
        return pdn_out, ae_out


gpu = torch.cuda.is_available()

autoencoder = AutoEncoder()
teacher = (384)
student = get_pdn_m(768)

autoencoder = autoencoder.eval()
teacher = teacher.eval()
student = student.eval()

if gpu:
    autoencoder.half().cuda()
    teacher.half().cuda()
    student.half().cuda()

quant_mult = torch.e
quant_add = torch.pi
with torch.no_grad():
    times = []
    for rep in range(2000):
        image = torch.randn(1, 3, 256, 256, dtype=torch.float16 if gpu else torch.float32)
        start = time()
        if gpu:
            image = image.cuda()

        t = teacher(image)
        s = student(image)

        st_map = torch.mean((t - s[:, :384]) ** 2, dim=1)
        ae = autoencoder(image)
        ae_map = torch.mean((ae - s[:, 384:]) ** 2, dim=1)
        st_map = st_map * quant_mult + quant_add
        ae_map = ae_map * quant_mult + quant_add
        result_map = st_map + ae_map
        result_on_cpu = result_map.cpu().numpy()
        timed = time() - start
        times.append(timed)
print(np.mean(times[-1000:]))#the mean runtime of the following 1000 forward passes