import torch
from torch import nn
import torch.nn.functional as F
import math

class Se_layer(nn.Module):
    
    def __init__(self, planes):
        super(Se_layer, self).__init__()

        self.expansion = 1

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,bias=False)
        self.bn_down = nn.BatchNorm2d(planes * self.expansion)

        self.act_f = nn.LeakyReLU(0.2,inplace=True)
        self.conv_up = nn.Conv2d(planes * self.expansion, planes, kernel_size=1,bias=False)
        self.bn_up = nn.BatchNorm2d(planes)

        self.sig = nn.Sigmoid()

    def forward(self,input):

        out = input

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.bn_down(out1)
        out1 = self.act_f(out1)

        out1 = self.conv_up(out1)
        out1 = self.bn_up(out1)        
        ratio = self.sig(out1)

        output = ratio * out

        return output,ratio
