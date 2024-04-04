import torch
from torch import nn
import torch.nn.functional as F

from model.non_local_embedded_gaussian import NONLocalBlock2D
from model.se_block import Se_layer
from model.joint_att_block import Weight_core1


class L_Block1(nn.Module):

    def __init__(self, in_planes, planes,stride=1,joint_dim=22):
        super(L_Block1, self).__init__()

        inteplanes = planes

        self.conv1 = nn.Conv2d(in_planes, inteplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inteplanes)

        self.conv2 = nn.Conv2d(inteplanes, inteplanes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inteplanes)

        self.conv2_1 = nn.Conv2d(inteplanes, inteplanes, kernel_size=(1,3),stride=stride, padding=(0,1), bias=False)
        self.bn2_1 = nn.BatchNorm2d(inteplanes)

        self.conv3 = nn.Conv2d(3*inteplanes, planes,kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.act_f = nn.LeakyReLU(0.2,inplace=True)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shorcut = True
            self.shortcut_conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            self.shortout_bn =nn.BatchNorm2d(planes)

        else:
            self.shorcut = False

        self.se_layer = Se_layer(3*inteplanes)

        # Non-local
        self.nonloc = NONLocalBlock2D(in_channels = inteplanes)

        # self.cam = CAM_Module1()
        self.core = Weight_core1(in_dim=joint_dim,time_dim=inteplanes)


    def forward(self, x, joint_att=0,act=None):

        # bottlenet 第一个1*1
        # x ：[B, 64, 22, 64]
        # x_0 ：[B, 64, 22, 64]
        x_0 = self.act_f(self.bn1(self.conv1(x)))

        # Feature normalization + Multi-head self-attention Unit
        # joint_att: [B,64,22,22]
        joint_att = self.core(x_0,act=act)

        # LIE
        # F distant
        # x_1_0 ：[B, 64, 22, 64]
        x_1_0 = self.act_f(self.nonloc(x_0)[0])
        # F adjacent
        # x_1_1：[B, 64, 22, 64]
        x_1_1 = self.act_f(self.bn2(self.conv2(x_0)))

        # GCE
        # x_1_2: [B, 64, 22, 64]      
        x_1_2_before = self.act_f(self.bn2_1(self.conv2_1(x_0)))
        # x_1_2: [B, 64, 22, 64] 
        x_1_2 = torch.matmul(joint_att,x_1_2_before)

        # AFFM 
        # x_1: [B, 64*3, 22, 64] 
        x_1 = torch.cat([x_1_0, x_1_1, x_1_2], 1)
        # x_2: [B, 64*3, 22, 64] 
        x_2,ratio = self.se_layer(x_1)

        # bottlenet 最后一个1*1
        # x_2: [B, 64, 22, 64] 
        x_2 = self.bn3(self.conv3(x_2))

        if self.shorcut == True:
            x_2 = x_2 + self.shortout_bn(self.shortcut_conv(x))
        else:
            x_2 = x_2 + x
        
        # x_2: [B, 64, 22, 64] 
        x_2 = self.act_f(x_2)

        return x_2,ratio


