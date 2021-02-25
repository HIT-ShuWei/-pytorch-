import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        #定义残差块
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            #为了使shortcut能够和output正常相加，需要控制维度相同，需要用1*1卷积层转换
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, input):
        output = self.left(input)
        output += self.shortcut(input)
        output = F.relu(output)

        return output

class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(64, 64, 2, stride=1)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7)
        self.flatten_layer = FlattenLayer()
        self.fc = nn.Linear(512, num_classes, bias=False)

    def make_layer(self,in_channels, out_channels, num_blocks, stride):
        # 重复将多个ResBLock堆叠在一起
        layers = []
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            layers.append(ResBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, input):
        #完整的ResNet-18结构
        out  = self.conv1(input)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_avg_pool(out)
        out = self.flatten_layer(out)
        '''
        问题记录：这里的global_avg和flatten在forward里面如果直接使用avg_pool2d/.view会出现维度错误
        需要在__init__中单独创建一些nn.Module的类，然后再forward中使用才能正常继续，否则就一直报错
        具体原因还没有找到
        '''
        output = self.fc(out)
        return output


def debug():
    net = ResNet(num_classes=2)
    # print(net)
    # 测试完整的网络结构
    X = torch.rand((1, 3, 224, 224))
    for name, layer in net.named_children():
        X = layer(X)
        print(name, ' output shape:\t', X.shape)

if __name__ == '__main__':
    debug()