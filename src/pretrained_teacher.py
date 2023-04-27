import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from myad.src.mainnet import TeacherNet
import torch.nn as nn
#device=torch.device("cpu")
class AnomalyDataset(Dataset):
    '''Anomaly detection dataset.
    - root_dir: path to the dataset to train the model on, eg: <path>/data/carpet
    - transform: list of transformation to apply on input image, eg: Resize, Normalize, etc
    - gt_transform: list of transformation to apply on gt image, eg: Resize.
    - constraint: filter to apply on the reading of the CSV file, a filter is a kwarg.
                  eg: type='train' to filter train data
                      label=0 to filter on anomaly-free data
    '''

    def __init__(self, root_dir, transform=transforms.ToTensor(), gt_transform=transforms.ToTensor(), **constraint):
        super(AnomalyDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_dir = os.path.join(self.root_dir, 'img')
        self.gt_dir = os.path.join(self.root_dir, 'ground_truth')
        self.dataset = self.root_dir.split('/')[-1]
        self.csv_file = os.path.join(self.root_dir, 'carpet' + '.csv')
        self.frame_list = self._get_dataset(self.csv_file, constraint)

    def _get_dataset(self, csv_file, constraint):
        '''Apply filter based on the constraint dict on the dataset'''
        df = pd.read_csv(csv_file, keep_default_na=False)
        df = df.loc[(df[list(constraint)] == pd.Series(constraint)).all(axis=1)]
        return df

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        item = self.frame_list.iloc[idx]
        img_path = os.path.join(self.img_dir, item['image_name'])
        label = item['label']
        image = Image.open(img_path).convert('RGB')

        if item['gt_name']:
            gt_path = os.path.join(self.gt_dir, item['gt_name'])
            gt = Image.open(gt_path).convert('L')
        else:
            gt = Image.new('L', image.size, color=0)

        sample = {'label': label}

        if self.transform:
            sample['image'] = self.transform(image)

        if self.gt_transform:
            sample['gt'] = self.gt_transform(gt)

        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = AnomalyDataset(root_dir='/media/densogup-1/8T/jyp/myad/data/carpet/',
                             transform=transforms.Compose([
                                 transforms.Resize((256, 256)),
                                 transforms.RandomCrop((256, 256)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                             type='train',
                             label=0)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    teacher = TeacherNet()
    teacher = teacher.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.0001, weight_decay=0.00001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for i, sample in enumerate(dataloader):
            images = sample['image'].cuda()
            labels = sample['label'].cuda()

            optimizer.zero_grad()
            outputs = teacher(images)
            labels = labels.to(outputs.dtype)
            loss = criterion(outputs, labels)  # convert labels to float
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss:{loss.item()}')

    torch.save(teacher.state_dict(), 'teacher_model.pth')