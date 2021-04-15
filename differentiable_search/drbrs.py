import torch
import torch.nn as nn
import torch.nn.functional as F

import math

RATIOS = [1,2,3,4,5,6,7,8]

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    def __init__(self, ch_in, ch_out, ratio, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv3x3(ch_in, ch_out//ratio, stride)
        self.bn1 = nn.BatchNorm2d(ch_out//ratio)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(ch_out//ratio, ch_out)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.downsample = None
        if stride != 1 or ch_in != ch_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MixedRatio(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(MixedRatio, self).__init__()
        self.bs = nn.ModuleList()
        for ratio in RATIOS:
            self.bs.append(Bottleneck(ch_in, ch_out, ratio, stride))

    def forward(self, x, alpha):
        softmax_alpha = F.softmax(alpha, dim=-1)
        return sum(a * b(x) for a, b in zip(softmax_alpha, self.bs))

class Cell(nn.Module):
    def __init__(self, ch_in, stride=1):
        super(Cell, self).__init__()
        self.ch_out = (ch_in) if ch_in==64 else (ch_in * 2)

        self.mixedratio1 = MixedRatio(ch_in, self.ch_out, stride)
        self.mixedratio2 = MixedRatio(self.ch_out, self.ch_out)

    def forward(self, x, alpha):
        x = self.mixedratio1(x, alpha)
        x = self.mixedratio2(x, alpha)
        return x

class DRBRS(nn.Module):
    def __init__(self, num_classes=10):
        super(DRBRS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cell1 = Cell(64)
        self.cell2 = Cell(self.cell1.ch_out, stride=2)
        self.cell3 = Cell(self.cell2.ch_out, stride=2)
        self.cell4 = Cell(self.cell3.ch_out, stride=2)

        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(self.cell4.ch_out, num_classes)

        self.alpha1 = nn.Parameter(1e-3 * torch.randn(len(RATIOS)), requires_grad=True)
        self.alpha2 = nn.Parameter(1e-3 * torch.randn(len(RATIOS)), requires_grad=True)
        self.alpha3 = nn.Parameter(1e-3 * torch.randn(len(RATIOS)), requires_grad=True)
        self.alpha4 = nn.Parameter(1e-3 * torch.randn(len(RATIOS)), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.cell1(x, self.alpha1)
        x = self.cell2(x, self.alpha2)
        x = self.cell3(x, self.alpha3)
        x = self.cell4(x, self.alpha4)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def arch_parameters(self):
        return [
            self.alpha1,
            self.alpha2,
            self.alpha3,
            self.alpha4
        ]

    def flatten_arch_parameters(self):
        return [
            *F.softmax(self.alpha1, dim=-1).tolist(),
            *F.softmax(self.alpha2, dim=-1).tolist(),
            *F.softmax(self.alpha3, dim=-1).tolist(),
            *F.softmax(self.alpha4, dim=-1).tolist()
        ]

    def current_arch(self):
        return [
            RATIOS[torch.argmax(self.alpha1).item()],
            RATIOS[torch.argmax(self.alpha2).item()],
            RATIOS[torch.argmax(self.alpha3).item()],
            RATIOS[torch.argmax(self.alpha4).item()]
        ]

    def current_arch_confidence(self):
        return [
            torch.max(F.softmax(self.alpha1, dim=-1)).item() * 100,
            torch.max(F.softmax(self.alpha2, dim=-1)).item() * 100,
            torch.max(F.softmax(self.alpha3, dim=-1)).item() * 100,
            torch.max(F.softmax(self.alpha4, dim=-1)).item() * 100,
        ]