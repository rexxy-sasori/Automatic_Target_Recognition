"""
Author: Rex Geng

neural network models that will be used directly by the main script
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn import modules as modulesc


class NNModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NNModel, self).__init__(*args, **kwargs)

    def forward(self, x):
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class ATRLite(nn.Module):
    def __init__(self, fmsize=32, m=5):
        super(ATRLite, self).__init__()
        self.fmsize = fmsize
        self.m = m

        self.conv1DW = nn.Conv2d(1, m, 5, bias=False)
        self.bn1DW = nn.BatchNorm2d(m)
        self.relu1DW = nn.ReLU(inplace=True)
        self.conv1PW = nn.Conv2d(m, fmsize, 1, bias=False)
        self.bn1PW = nn.BatchNorm2d(fmsize)
        self.relu1PW = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(fmsize, 2 * fmsize, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(fmsize * 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(2 * fmsize, 4 * fmsize, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(fmsize * 4)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(4 * fmsize, 16, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1DW(x)
        x = self.bn1DW(x)
        x = self.relu1DW(x)

        x = self.conv1PW(x)
        x = self.bn1PW(x)
        x = self.relu1PW(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'dbknetlite_fm{}_m{}.pt.tar'.format(self.fmsize, self.m)


class ATRLiteC0F0(nn.Module):
    def __init__(self, fmsize=1, m=5):
        super(ATRLiteC0F0, self).__init__()
        self.fmsize = fmsize
        self.m = m

        self.conv1DW = nn.Conv2d(1, m, 5, bias=False)
        self.bn1DW = nn.BatchNorm2d(m)
        self.relu1DW = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.conv1DW(x)
        x = self.bn1DW(x)
        x = self.relu1DW(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'dbknetlite_only_c0_f0_fm{}_m{}.pt.tar'.format(self.fmsize, self.m)


class BASE(nn.Module):
    def __init__(self, fmsize=32, K=5, R=128):
        super(BASE, self).__init__()

        self.fmsize = fmsize
        self.K = K  # kernel size of the first layer
        self.R = R  # size of the input image

        self.conv1 = nn.Conv2d(1, fmsize, K, bias=False)
        self.bn1 = nn.BatchNorm2d(fmsize)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(fmsize * 1, 2 * fmsize, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(fmsize * 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(2 * fmsize, 4 * fmsize, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(fmsize * 4)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(4 * fmsize, 16, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)

        d_in = {'32': 16, '64': 400, '128': 2704}
        self.d_in = d_in[str(self.R)]
        self.fc1 = nn.Linear(self.d_in, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # print(x.shape)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # print(x.shape)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = x.view(x.size()[0], -1)
        # print(x.shape)
        # assert(1==2)
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'basenet_fm{}_k{}_r{}.pt.tar'.format(self.fmsize, self.K, self.R)


class BASEDWS(nn.Module):
    def __init__(self, fmsize=32, K=9, R=128):
        super(BASEDWS, self).__init__()

        self.fmsize = fmsize
        self.K = K  # kernel size of the first layer
        self.R = R  # size of the input image

        self.conv1DW = nn.Conv2d(1, 1, K, bias=False)
        self.bn1DW = nn.BatchNorm2d(1)
        self.relu1DW = nn.ReLU(inplace=True)
        self.conv1PW = nn.Conv2d(1, fmsize * 1, 1, bias=False)
        self.bn1PW = nn.BatchNorm2d(fmsize * 1)
        self.relu1PW = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(fmsize * 1, 2 * fmsize, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(fmsize * 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(2 * fmsize, 4 * fmsize, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(fmsize * 4)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(4 * fmsize, 16, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)

        d_in = {'32': 16, '64': 400, '128': 2704}
        self.d_in = d_in[str(self.R)]
        self.fc1 = nn.Linear(self.d_in, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1DW(x)
        x = self.bn1DW(x)
        x = self.relu1DW(x)

        x = self.conv1PW(x)
        x = self.bn1PW(x)
        x = self.relu1PW(x)

        # print(x.shape)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # print(x.shape)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = x.view(x.size()[0], -1)
        # print(x.shape)
        # assert(1==2)
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'basedwsnet_fm{}_k{}_r{}.pt.tar'.format(self.fmsize, self.K, self.R)


class DingNet(nn.Module):
    def __init__(self):
        super(DingNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, 3)  # 96 126 126
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(96, 96, 3)  # 96 124 124
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(96, 256, 3)  # 256 60 60
        self.relu3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.D_in = 30 * 30 * 256
        self.fc1 = nn.Linear(self.D_in, 1000)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, 16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool2(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        return x

    def __str__(self):
        return 'dingnet.pt.tar'


class ChenNet(nn.Module):
    def __init__(self):
        super(ChenNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 6)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 5)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(128, 10, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = x.view(x.size()[0], -1)
        return x

    def __str__(self):
        return 'chennet.pt.tar'


class GaoNet(nn.Module):
    def __init__(self):
        super(GaoNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 18, 9)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(7, 7)

        self.conv2 = nn.Conv2d(18, 120, 5)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(120, 120, 2)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(120, 10, 1)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'gaonet.pt.tar'


class WagnerNet(nn.Module):
    def __init__(self):
        super(WagnerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 13)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(20, 120, 13)
        self.fc = nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 120)
        x = self.fc(x)
        return x

    def __str__(self):
        return 'wagnernet.pt.tar'


class WagnerNet_rev(nn.Module):
    def __init__(self, m=3, f=30, dropout_rate=0):
        self.m = m
        self.f = f
        super(WagnerNet_rev, self).__init__()
        self.conv1DW = nn.Conv2d(1, m, 13)
        self.conv1PW = nn.Conv2d(m, 20, 1)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.drop_conv = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(20, f, 13)
        self.drop_fc = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(f, 10)

    def forward(self, x):
        x = F.relu(self.conv1DW(x))
        x = F.relu(self.conv1PW(x))
        x = self.pool1(x)
        x = self.drop_conv(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.f)
        x = self.drop_fc(x)
        x = self.fc(x)
        return x

    def __str__(self):
        return 'wagnernet_rev_f%d_m%d.pt.tar' % (self.f, self.m)


class MorganNet(nn.Module):
    def __init__(self):
        super(MorganNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 18, 9)  # 18 120 12
        self.pool1 = nn.MaxPool2d(6, 6)  # 18 20 20
        self.conv2 = nn.Conv2d(18, 36, 5)  # 36 16 16
        self.pool2 = nn.MaxPool2d(4, 4)  # 36 4 4
        self.conv3 = nn.Conv2d(36, 120, 4)  # 120 1 1
        self.fc = nn.Linear(120, 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = self.fc(x)
        return x

    def __str__(self):
        return 'morgannet.pt.tar'


class ATRLite48(ATRLite):
    def __init__(self, fmsize=32, m=5):
        super(ATRLite48, self).__init__(fmsize=fmsize, m=m)
        self.input_pool = nn.AdaptiveAvgPool2d(32)

    def forward(self, x):
        assert x.shape[2] == 48
        x = self.input_pool(x)
        return super(ATRLite48, self).forward(x)

    def __str__(self):
        return 'dbknetlite48_fm{}_m{}.pt.tar'.format(self.fmsize, self.m)


class ATRLite64(ATRLite):
    def __init__(self, fmsize=8, m=5):
        super(ATRLite64, self).__init__(fmsize=fmsize, m=m)
        self.input_pool = nn.AdaptiveAvgPool2d(32)

    def forward(self, x):
        assert x.shape[2] == 64
        x = self.input_pool(x)
        return super(ATRLite64, self).forward(x)

    def __str__(self):
        return 'dbknetlite64_fm{}_m{}.pt.tar'.format(self.fmsize, self.m)


class ATRLite96(ATRLite):
    def __init__(self, fmsize=32, m=5):
        super(ATRLite96, self).__init__()
        self.input_pool = nn.AdaptiveAvgPool2d(32)

    def forward(self, x):
        x = self.input_pool(x)
        return super(ATRLite96, self).forward(x)

    def __str__(self):
        return 'dbknetlite96_fm{}_m{}.pt.tar'.format(self.fmsize, self.m)


class ATRLite48_efc(nn.Module):
    def __init__(self, fmsize=32, m=5, dropout_rate=0):
        super(ATRLite48_efc, self).__init__()
        self.fmsize = fmsize
        self.m = m
        self.dropout_rate = dropout_rate

        self.conv1DW = nn.Conv2d(1, m, 5, bias=False)
        self.bn1DW = nn.BatchNorm2d(m)
        self.relu1DW = nn.ReLU(inplace=True)
        self.conv1PW = nn.Conv2d(m, fmsize, 1, bias=False)
        self.bn1PW = nn.BatchNorm2d(fmsize)
        self.relu1PW = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(fmsize, 2 * fmsize, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(fmsize * 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(2 * fmsize, 4 * fmsize, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(fmsize * 4)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop_conv4 = nn.Dropout(dropout_rate)

        self.conv4 = nn.Conv2d(4 * fmsize, 16, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)

        self.drop_fc = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(9 * 16, 10)

    def forward(self, x):
        x = self.conv1DW(x)
        x = self.bn1DW(x)
        x = self.relu1DW(x)

        x = self.conv1PW(x)
        x = self.bn1PW(x)
        x = self.relu1PW(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.drop_conv4(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = x.view(x.size()[0], -1)
        x = self.drop_fc(x)
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'dbknetlite48_efc_fm{}_m{}_d{}.pt.tar'.format(self.fmsize, self.m, self.dropout_rate)


class ATRLite64_efc(nn.Module):
    def __init__(self, fmsize=32, m=5, dropout_rate=0):
        super(ATRLite64_efc, self).__init__()
        self.fmsize = fmsize
        self.m = m
        self.dropout_rate = dropout_rate

        self.conv1DW = nn.Conv2d(1, m, 5, bias=False)
        self.bn1DW = nn.BatchNorm2d(m)
        self.relu1DW = nn.ReLU(inplace=True)
        self.conv1PW = nn.Conv2d(m, fmsize, 1, bias=False)
        self.bn1PW = nn.BatchNorm2d(fmsize)
        self.relu1PW = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(fmsize, 2 * fmsize, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(fmsize * 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(2 * fmsize, 4 * fmsize, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(fmsize * 4)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop_conv4 = nn.Dropout(dropout_rate)

        self.conv4 = nn.Conv2d(4 * fmsize, 16, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)

        self.drop_fc = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(25 * 16, 10)

    def forward(self, x):
        x = self.conv1DW(x)
        x = self.bn1DW(x)
        x = self.relu1DW(x)

        x = self.conv1PW(x)
        x = self.bn1PW(x)
        x = self.relu1PW(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.drop_conv4(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = x.view(x.size()[0], -1)
        x = self.drop_fc(x)
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'dbknetlite64_efc_fm{}_m{}_d{}.pt.tar'.format(self.fmsize, self.m, self.dropout_rate)


class ATRLite80_efc(nn.Module):
    def __init__(self, fmsize=32, m=5, dropout_rate=0):
        super(ATRLite80_efc, self).__init__()
        self.fmsize = fmsize
        self.m = m
        self.dropout_rate = dropout_rate

        self.conv1DW = nn.Conv2d(1, m, 5, bias=False)
        self.bn1DW = nn.BatchNorm2d(m)
        self.relu1DW = nn.ReLU(inplace=True)
        self.conv1PW = nn.Conv2d(m, fmsize, 1, bias=False)
        self.bn1PW = nn.BatchNorm2d(fmsize)
        self.relu1PW = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(fmsize, 2 * fmsize, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(fmsize * 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(2 * fmsize, 4 * fmsize, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(fmsize * 4)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop_conv4 = nn.Dropout(dropout_rate)

        self.conv4 = nn.Conv2d(4 * fmsize, 16, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)

        self.drop_fc = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(49 * 16, 10)

    def forward(self, x):
        x = self.conv1DW(x)
        x = self.bn1DW(x)
        x = self.relu1DW(x)

        x = self.conv1PW(x)
        x = self.bn1PW(x)
        x = self.relu1PW(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.drop_conv4(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = x.view(x.size()[0], -1)
        x = self.drop_fc(x)
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'dbknetlite80_efc_fm{}_m{}_d{}.pt.tar'.format(self.fmsize, self.m, self.dropout_rate)


class ATRLite64_isk(nn.Module):
    def __init__(self, fmsize=32, m=5, dropout_rate=0):
        super(ATRLite64_isk, self).__init__()
        self.fmsize = fmsize
        self.m = m
        self.dropout_rate = dropout_rate

        self.conv1DW = nn.Conv2d(1, m, 13, bias=False)
        self.bn1DW = nn.BatchNorm2d(m)
        self.relu1DW = nn.ReLU(inplace=True)
        self.conv1PW = nn.Conv2d(m, fmsize, 1, bias=False)
        self.bn1PW = nn.BatchNorm2d(fmsize)
        self.relu1PW = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(fmsize, 2 * fmsize, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(fmsize * 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(2 * fmsize, 4 * fmsize, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(fmsize * 4)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop_conv4 = nn.Dropout(dropout_rate)

        self.conv4 = nn.Conv2d(4 * fmsize, 16, 5, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)

        self.drop_fc = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1DW(x)
        x = self.bn1DW(x)
        x = self.relu1DW(x)

        x = self.conv1PW(x)
        x = self.bn1PW(x)
        x = self.relu1PW(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.drop_conv4(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = x.view(x.size()[0], -1)
        x = self.drop_fc(x)
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'dbknetlite64_isk_fm{}_m{}_d{}.pt.tar'.format(self.fmsize, self.m, self.dropout_rate)


class ATRLite80_isk(nn.Module):
    def __init__(self, fmsize=32, m=5, dropout_rate=0):
        super(ATRLite80_isk, self).__init__()
        self.fmsize = fmsize
        self.m = m
        self.dropout_rate = dropout_rate

        self.conv1DW = nn.Conv2d(1, m, 13, bias=False)
        self.bn1DW = nn.BatchNorm2d(m)
        self.relu1DW = nn.ReLU(inplace=True)
        self.conv1PW = nn.Conv2d(m, fmsize, 1, bias=False)
        self.bn1PW = nn.BatchNorm2d(fmsize)
        self.relu1PW = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(fmsize, 2 * fmsize, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(fmsize * 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(2 * fmsize, 4 * fmsize, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(fmsize * 4)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop_conv4 = nn.Dropout(dropout_rate)

        self.conv4 = nn.Conv2d(4 * fmsize, 16, 7, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)

        self.drop_fc = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1DW(x)
        x = self.bn1DW(x)
        x = self.relu1DW(x)

        x = self.conv1PW(x)
        x = self.bn1PW(x)
        x = self.relu1PW(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.drop_conv4(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = x.view(x.size()[0], -1)
        x = self.drop_fc(x)
        x = self.fc1(x)
        return x

    def __str__(self):
        return 'dbknetlite80_isk_fm{}_m{}_d{}.pt.tar'.format(self.fmsize, self.m, self.dropout_rate)


class GuoCapSAREncoder(nn.Module):
    def __init__(self):
        super(GuoCapSAREncoder, self).__init__()
        self.feautures = nn.Sequential(
            nn.Conv2d(1, 128, 9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 64, 5),
            nn.Dropout(0.5),
        )

        self.cap = modulesc.CapsuleLayer(
            num_in_channel=64,
            num_out_channel=128,
            kernel_size=8,
            stride=2,
            num_primary_cap=1024,
            num_sar_cap=10,
            input_dim=32
        )

    def forward(self, x):
        before_cap = self.feautures(x)
        likelihood = self.cap(before_cap)
        return likelihood

    def __str__(self):
        return 'guo_capsule_primary.pt.tar'

    def __repr__(self):
        return 'guo_capsule_primary.pt.tar'


class YangCapSAREncoder(nn.Module):
    def __init__(self):
        super(YangCapSAREncoder, self).__init__()
        self.feautures = nn.Sequential(
            nn.Conv2d(1, 16, 5, dilation=2, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, dilation=2, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5, dilation=2, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, dilation=2, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 5, dilation=2, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )  # output size BatchSizex256x8x8

        self.cap = modulesc.CapsuleLayer(
            num_in_channel=256,
            num_out_channel=256,
            kernel_size=5,
            stride=2,
            num_primary_cap=128,
            num_sar_cap=10,
            input_dim=8,
        )

    def forward(self, x):
        before_cap = self.feautures(x)
        likelihood = self.cap(before_cap)
        return likelihood

    def __str__(self):
        return 'yang_capsule_primary.pt.tar'


class ShahCapSAREncoder(nn.Module):
    def __init__(self):
        super(ShahCapSAREncoder, self).__init__()
        self.feautures = nn.Sequential(
            nn.Conv2d(1, 256, 9),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 5),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.cap = modulesc.CapsuleLayer(
            num_in_channel=256,
            num_out_channel=256,
            kernel_size=9,
            stride=2,
            num_primary_cap=32 * 56 * 56,
            num_sar_cap=10,
            input_dim=8
        )

    def forward(self, x):
        before_cap = self.feautures(x)
        likelihood = self.cap(before_cap)
        return likelihood

    def __str__(self):
        return 'shah_capsule_primary.pt.tar'


class ZhangCapSAREncoder(nn.Module):
    def __init__(self):
        super(ZhangCapSAREncoder, self).__init__()
        self.feautures = nn.Sequential(
            nn.Conv2d(1, 16, 9),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 9),
            nn.ReLU(inplace=True),
        )

        self.cap = modulesc.CapsuleLayer(
            num_in_channel=32,
            num_out_channel=64,
            kernel_size=9,
            stride=2,
            num_primary_cap=400,
            num_sar_cap=10,
            input_dim=64
        )

    def forward(self, x):
        before_cap = self.feautures(x)
        likelihood = self.cap(before_cap)
        return likelihood

    def __str__(self):
        return 'zhang_capsule_primary.pt.tar'


__MODELS__ = {
    'atrlite': ATRLite,
    'atrlite_c0_f0': ATRLiteC0F0,
    'dingnet': DingNet,
    'chennet': ChenNet,
    'gaonet': GaoNet,
    'wagnernet': WagnerNet,
    'wagnernet_rev': WagnerNet_rev,
    'morgan': MorganNet,
    'atrlite48': ATRLite48,
    'atrlite64': ATRLite64,
    'atrlite96': ATRLite96,
    'atrlite48_efc': ATRLite48_efc,
    'atrlite64_efc': ATRLite64_efc,
    'atrlite80_efc': ATRLite80_efc,
    'atrlite64_isk': ATRLite64_isk,
    'atrlite80_isk': ATRLite80_isk,
    'guo_capsule': GuoCapSAREncoder,
    'yang_cap': YangCapSAREncoder,
    'shah_cap': ShahCapSAREncoder,
    'zhang_cap': ZhangCapSAREncoder,
}

__MODELS_INPUTS__ = {
    ATRLite: (torch.randn(1, 1, 32, 32)),
    ATRLiteC0F0: (torch.randn(1, 1, 32, 32)),
    DingNet: (torch.randn(1, 1, 128, 128)),
    ChenNet: (torch.randn(1, 1, 88, 88)),
    GaoNet: (torch.randn(1, 1, 64, 64)),
    WagnerNet: (torch.randn(1, 1, 64, 64)),
    WagnerNet_rev: (torch.randn(1, 1, 64, 64)),
    MorganNet: (torch.randn(1, 1, 128, 128)),
    ATRLite48: (torch.randn(1, 1, 48, 48)),
    ATRLite64: (torch.randn(1, 1, 64, 64)),
    ATRLite96: (torch.randn(1, 1, 96, 96)),
    ATRLite48_efc: (torch.randn(1, 1, 48, 48)),
    ATRLite64_efc: (torch.randn(1, 1, 64, 64)),
    ATRLite80_efc: (torch.randn(1, 1, 80, 80)),
    ATRLite64_isk: (torch.randn(1, 1, 64, 64)),
    ATRLite80_isk: (torch.randn(1, 1, 80, 80)),
    GuoCapSAREncoder: (torch.randn(1, 1, 92, 92)),
    YangCapSAREncoder: (torch.randn(1, 1, 64, 64)),
    ShahCapSAREncoder: (torch.randn(1, 1, 128, 128)),
    ZhangCapSAREncoder: (torch.randn(1, 1, 64, 64))
}
