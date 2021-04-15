import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BottleneckBlock(nn.Module):
    def __init__(self, inplanes, outplanes, ratio, stride=1, downsample=None):
        sqzdplanes = (inplanes // ratio) if inplanes==64 else (inplanes * 2 // ratio)

        super(BottleneckBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, sqzdplanes, stride)
        self.bn1 = nn.BatchNorm2d(sqzdplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(sqzdplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = downsample

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

class ResNet(nn.Module):
    design = [2, 2, 2, 2]

    def __init__(self, ratio=[1, 1, 1, 1], num_classes=10):
        self.inplanes = 64

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ratio[0], ResNet.design[0])

        self.layer2 = self._make_layer(ratio[1], ResNet.design[1], stride=2)

        self.layer3 = self._make_layer(ratio[2], ResNet.design[2], stride=2)

        self.layer4 = self._make_layer(ratio[3], ResNet.design[3], stride=2)

        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, ratio, blocks, stride=1):
        outplanes = (self.inplanes) if self.inplanes==64 else (self.inplanes * 2)

        downsample = None
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        layers = []
        layers.append(BottleneckBlock(self.inplanes, outplanes, ratio, stride, downsample))
        self.inplanes = outplanes
        for i in range(1, blocks):
            layers.append(BottleneckBlock(self.inplanes, outplanes, ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x