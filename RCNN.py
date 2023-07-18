import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# ————————————————————————————————仅使用Gabor滤波的实部———————————————————————————————————————

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        """
        tex_path : txt文本路径，该文本包含了图像的路径信息，以及标签信息
        transform：数据处理，对图像进行随机剪裁，以及转换成tensor
        """
        fh = open(txt_path, 'r')  # 读取文件
        imgs = []  # 用来存储路径与标签
        # 一行一行的读取
        for line in fh:
            line = line.rstrip()  # 这一行就是图像的路径，以及标签
            words = line.split()
            imgs.append((words[0], words[1]))  # 路径和标签添加到列表中
            self.imgs = imgs
            self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]  # 通过index索引返回一个图像路径fn 与 标签label
        label = float(label)
        img = cv2.imread(fn, 1)
        img = cv2.resize(img, (224, 224))  # 图像尺寸为256*256
        if self.transform is not None:
            img = self.transform(img)
        label = torch.as_tensor(label)
        return img, label  # 这就返回一个样本

    def __len__(self):
        return len(self.imgs)  # 返回长度，index就会自动的指导读取多少

class CConv(nn.Module):
    def __init__(self, kernel_size=5, in_channels=3, out_channel=32):
        super(CConv, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(16,32,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.feature(x)
        x = self.avg(x)
        feature = torch.flatten(x, 1)
        out = self.fc1(feature)
        return feature,out

def train(log_interval, model, device, train_loader, optimizer, epoch,list):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        _, output = model(data)
        loss = F.mse_loss(output.to(torch.float32).squeeze(-1), target.to(torch.float32)).cuda()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            list.append(loss.item())
    return list

def test(model, device, test_loader):
    model.eval()
    mae = 0
    rmse = 0
    with torch.no_grad():
        count = 1
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            _, output = model(data)
            mseloss = F.mse_loss(output.to(torch.float32).squeeze(-1), target.to(torch.float32))
            maeloss = mean_absolute_error(target.to(torch.float32).cpu(), output.to(torch.float32).squeeze(-1).cpu())
            rmse += mseloss**0.5
            mae += maeloss
            count += 1

    # print('\nTest mseloss: {:.4f}\n'.format(mseloss/count))
    print('\nTest rmse: {:.4f}\n'.format(rmse/count))
    print('Test mae: {:.4f}\n'.format(mae/count))

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batchsize = 32
    epochs = 50
    log_interval = 10
    save_model = True
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5)])  # 转换
    torch.manual_seed(42)  # 随机种子
    data = MyDataset("../dataset/p55_train.txt", transform=transform)
    trainsize = int(len(data) * 0.8)
    testsize = len(data) - trainsize
    train_data, test_data = torch.utils.data.random_split(data, [trainsize, testsize])
    train_loader = DataLoader(train_data, batchsize, shuffle=True)
    test_loader = DataLoader(test_data, batchsize, shuffle=True)

    model = CConv().cuda()
    fe = list(map(id, model.feature.parameters()))  # 返回的是parameters的 内存地址
    f1 = list(map(id, model.fc1.parameters()))  # 返回的是parameters的 内存地址
    base_params = filter(lambda p: id(p) not in fe + f1, model.parameters())
    optimizer = optim.Adam([
        {'params': base_params},
        {'params': model.feature.parameters(), 'lr': 0.001},
        {'params': model.fc1.parameters(), 'lr': 0.001},], 0.01)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.5)  # 动态学习率

    losslist = []
    for epoch in range(1, epochs + 1):
        a = train(log_interval, model, device, train_loader, optimizer, epoch, losslist)
        scheduler.step()
        test(model, device, test_loader)
        print('学习率:',scheduler.get_last_lr())

    if (save_model):
        torch.save(model.state_dict(),"../all_pt/Gabor_CA_model.pth")

if __name__ == '__main__':
    main()