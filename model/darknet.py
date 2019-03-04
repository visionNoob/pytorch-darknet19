from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from base import BaseModel

model_paths = {
    'darknet19': 'https://s3.ap-northeast-2.amazonaws.com/deepbaksuvision/darknet19-deepBakSu-e1b3ec1e.pth'
}

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


class Darknet19(BaseModel):
    def __init__(self, pretrained=True):
        super(Darknet19, self).__init__()

        self.features = nn.Sequential(OrderedDict([
        ('layer1', nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1_1', nn.BatchNorm2d(32)),
            ('leaky1_1', nn.LeakyReLU(0.1, inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))),
        ('layer2', nn.Sequential(OrderedDict([
            ('conv2_1', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2_1', nn.BatchNorm2d(64)),
            ('leaky2_1', nn.LeakyReLU(0.1, inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))),
        ('layer3', nn.Sequential(OrderedDict([
            ('conv3_1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn3_1', nn.BatchNorm2d(128)),
            ('leaky3_1', nn.LeakyReLU(0.1, inplace=True)),
            ('conv3_2', nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn3_2', nn.BatchNorm2d(64)),
            ('leaky3_2', nn.LeakyReLU(0.1, inplace=True)),
            ('conv3_3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn3_3', nn.BatchNorm2d(128)),
            ('leaky3_3', nn.LeakyReLU(0.1, inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))),
        ('layer4', nn.Sequential(OrderedDict([
            ('conv4_1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn4_1', nn.BatchNorm2d(256)),
            ('leaky4_1', nn.LeakyReLU(0.1, inplace=True)),
            ('conv4_2', nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn4_2', nn.BatchNorm2d(128)),
            ('leaky4_2', nn.LeakyReLU(0.1, inplace=True)),
            ('conv4_3', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn4_3', nn.BatchNorm2d(256)),
            ('leaky4_3', nn.LeakyReLU(0.1, inplace=True)),
            ('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))),
        ('layer5', nn.Sequential(OrderedDict([
            ('conv5_1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn5_1', nn.BatchNorm2d(512)),
            ('leaky5_1', nn.LeakyReLU(0.1, inplace=True)),
            ('conv5_2', nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn5_2', nn.BatchNorm2d(256)),
            ('leaky5_2', nn.LeakyReLU(0.1, inplace=True)),
            ('conv5_3', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn5_3', nn.BatchNorm2d(512)),
            ('leaky5_3', nn.LeakyReLU(0.1, inplace=True)),
            ('conv5_4', nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=1, bias=False)),
            ('bn5_4', nn.BatchNorm2d(256)),
            ('leaky5_4', nn.LeakyReLU(0.1, inplace=True)),
            ('conv5_5', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn5_5', nn.BatchNorm2d(512)),
            ('leaky5_5', nn.LeakyReLU(0.1, inplace=True)),
            ('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))),
        ('layer6', nn.Sequential(OrderedDict([
            ('conv6_1', nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn6_1', nn.BatchNorm2d(1024)),
            ('leaky6_1', nn.LeakyReLU(0.1, inplace=True)),
            ('conv6_2', nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn6_2', nn.BatchNorm2d(512)),
            ('leaky6_2', nn.LeakyReLU(0.1, inplace=True)),
            ('conv6_3', nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn6_3', nn.BatchNorm2d(1024)),
            ('leaky6_3', nn.LeakyReLU(0.1, inplace=True)),
            ('conv6_4', nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=1, bias=False)),
            ('bn6_4', nn.BatchNorm2d(512)),
            ('leaky6_4', nn.LeakyReLU(0.1, inplace=True)),
            ('conv6_5', nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn6_5', nn.BatchNorm2d(1024)),
            ('leaky6_5', nn.LeakyReLU(0.1, inplace=True))
        ])))
        ]))

        self.classifier = nn.Sequential(OrderedDict([
        ('conv7_1', nn.Conv2d(1024, 1000, kernel_size=(1, 1), stride=(1, 1))),
        ('globalavgpool', GlobalAvgPool2d()),
        ('softmax', nn.Softmax(dim=1))
        ]))

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_paths['darknet19'],  progress=True))
            print('Model is loaded')

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out
