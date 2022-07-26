import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_Block1(nn.Module):
    expansion = 1

    def __init__(self, input_channel, output_channel, stride):
        super(Residual_Block1, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1)
        self.batch1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(output_channel)

        self.origin_layer = nn.Sequential()
        if stride != 1 or input_channel != output_channel:
            self.origin_layer = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(output_channel))

    def forward(self, x):
        out = F.relu(self.batch1(self.conv1(x)))
        out = self.batch2(self.conv2(out))
        out += self.origin_layer(x)
        out = F.relu(out)

        return out

class Residual_Block2(nn.Module):
    expansion = 4

    def __init__(self, input_channel, output_channel, stride):
        super(Residual_Block2, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=1, padding=0)
        self.batch1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=stride, padding=1)
        self.batch2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel*self.expansion, kernel_size=1, padding=0)
        self.batch3 = nn.BatchNorm2d(output_channel*self.expansion)

        self.origin_layer = nn.Sequential()
        if stride != 1 or input_channel != (output_channel*self.expansion):
            self.origin_layer = nn.Sequential(
                nn.Conv2d(input_channel, output_channel*self.expansion, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(output_channel*self.expansion))

    def forward(self, x):
        out = F.relu(self.batch1(self.conv1(x)))
        out = F.relu(self.batch2(self.conv2(out)))
        out = self.batch3(self.conv3(out))
        out += self.origin_layer(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num, num_class=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.batch = nn.BatchNorm2d(64)
        self.block1 = self.make_layers(block, num[0], 64, 1)
        self.block2 = self.make_layers(block, num[1], 128, 2)
        self.block3 = self.make_layers(block, num[2], 256, 2)
        self.block4 = self.make_layers(block, num[3], 512, 2)
        self.fc = nn.Linear(512 * block.expansion, num_class)

    def forward(self, x):
        x = F.relu(self.batch(self.conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        feature = x
        x = self.fc(x)

        return x, feature

    def make_layers(self, block, num, channel, stride):
        strides = [stride] + [1] * (num - 1)
        layers = []
        for stride in strides:
            layers += [block(self.in_planes, channel, stride)]
            self.in_planes = channel * block.expansion

        return nn.Sequential(*layers)


class ResNet_cifar(nn.Module):
    def __init__(self, block, num, num_class=10):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.batch = nn.BatchNorm2d(16)
        self.block1 = self.make_layers(block, num[0], 16, 1)
        self.block2 = self.make_layers(block, num[1], 32, 2)
        self.block3 = self.make_layers(block, num[2], 64, 2)
        self.fc = nn.Linear(64 * block.expansion, num_class)

    def forward(self, x):
        x = F.relu(self.batch(self.conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        feature = x
        x = self.fc(x)

        return x, feature

    def make_layers(self, block, num, channel, stride):
        strides = [stride] + [1] * (num - 1)
        layers = []
        for stride in strides:
            layers += [block(self.in_planes, channel, stride)]
            self.in_planes = channel * block.expansion

        return nn.Sequential(*layers)



def ResNet18(num_class=10):
    return ResNet(Residual_Block1, [2, 2, 2, 2], num_class=num_class)


def ResNet34(num_class=10):
    return ResNet(Residual_Block1, [3, 4, 6, 3], num_class=num_class)


def ResNet50(num_class=10):
    return ResNet(Residual_Block2, [3, 4, 6, 3], num_class=num_class)


def ResNet101(num_class=10):
    return ResNet(Residual_Block2, [3, 4, 23, 3], num_class=num_class)


def ResNet152(num_class=10):
    return ResNet(Residual_Block2, [3, 8, 36, 3], num_class=num_class)


def ResNet32(num_class=10):
    return ResNet_cifar(Residual_Block1, [5, 5, 5], num_class=num_class)