import torch
from torch import nn
import torch.nn.functional as F
from torch import nn
from model.tcn import TemporalConvNet


class Multi_T2_abalation_test02(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, **kwargs):
        super(Multi_T2_abalation_test02,self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn0 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(1,3), stride=1, padding=(0,1),bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel, kernel_size=(1,5), stride=1, padding=(0,2),bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.conv4 = nn.Conv2d(in_channels=out_channel*3,out_channels=out_channel, kernel_size=1, stride=1, padding=0,bias=False)

        self.acf = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, input):

        # out_0 ： [B, 3, 22, 10]
        out_0 = input

        # Intra-jont features extration

        # out_0_0 ： [B, 64, 22, 10]
        out_0_0 = self.bn0(self.conv0(out_0))
        # out_0_2 ： [B, 64, 22, 10]
        out_0_2 = self.bn2(self.conv2(out_0))
        # out_0_3 ： [B, 64, 22, 10]
        out_0_3 = self.bn3(self.conv3(out_0))

        # out_all:  [B, 64*3, 22, 10]
        out_all = torch.cat([out_0_0, out_0_2, out_0_3], 1)
        out_0_4 = self.acf(out_all)

        # output:  [B, 64, 22, 10]
        output = self.conv4(out_0_4)

        return output





