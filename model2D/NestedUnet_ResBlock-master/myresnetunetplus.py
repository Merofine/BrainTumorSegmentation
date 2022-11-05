# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return self.down_sample(out),out

class ResNetUnetPlus(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        num_class = 3
        num_channels = 4
        nb_filter = [32, 64, 128, 256, 512]

        """
        Basebone
        """
        #resnet = models.resnet34(pretrained=False)
        self.resblock1 = ResBlock(num_channels,nb_filter[0])
        self.resblock2 = ResBlock(nb_filter[0], nb_filter[1])
        self.resblock3 = ResBlock(nb_filter[1], nb_filter[2])
        self.resblock4 = ResBlock(nb_filter[2], nb_filter[3])
        self.resblock5 = ResBlock(nb_filter[3], nb_filter[4])
        #self.firstconv = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        #self.firstbn = nn.BatchNorm2d(32)
        #self.firstrelu = resnet.relu
        #self.firstmaxpool = resnet.maxpool

        #self.encoder1 = resnet.layer1
        #self.encoder2 = resnet.layer2
        #self.encoder3 = resnet.layer3
        #self.encoder4 = resnet.layer4

        #self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #self.conv0_0 = VGGBlock(args.input_channels, nb_filter[0], nb_filter[0])
        #self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        #self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        #self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        #self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 3, kernel_size=1)


    def forward(self, input):
        x,x0_0 = self.resblock1(input)
        x,x1_0 = self.resblock2(x)
        x, x2_0 = self.resblock3(x)
        x, x3_0 = self.resblock4(x)
        _, x4_0 = self.resblock5(x)
        #print (x0_0.shape)32, 160, 160
        #print (x1_0.shape)64, 80, 80
        #print (x2_0.shape)128, 40, 40
        #print (x3_0.shape)256, 20, 20
        #print (x4_0.shape)512, 10, 10

        #input = self.firstconv(input)
        #input = self.firstbn(input)
        #input = F.relu(input)
        # Encoder
        #x = self.firstmaxpool(input)  # 32
        #print(x.shape)
        #e1 = self.encoder1(x)  # 64
        #e2 = self.encoder2(e1)  # 128
        #e3 = self.encoder3(e2)  # 256
        #e4 = self.encoder4(e3)  # 512


        #Decoder
        #x0_0 = input # 64, 160, 160
        #print(x0_0.shape)
        #x1_0 = e1 # 64, 80, 80
        #print(x1_0.shape)
        #out = torch.cat([x0_0, self.up(x1_0)], 1) # 128, 160, 160
        #print(out.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        #x2_0 = e2
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        #x3_0 = e3
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        #x4_0 = e4
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
