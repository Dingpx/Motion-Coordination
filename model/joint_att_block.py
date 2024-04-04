import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class Weight_core1(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim=22,time_dim=64):
        super(Weight_core1, self).__init__()

        self.inter_channel = in_dim

        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels=in_dim, out_channels=self.inter_channel, kernel_size=3,padding=1,bias=False),
            # nn.BatchNorm2d(self.inter_channel),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels=self.inter_channel,out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2,inplace=True)
            )
        self.query_conv = nn.Conv2d(in_channels=time_dim, out_channels=time_dim, kernel_size=1)

    def forward(self,input,act=None):

        # input ：[B, 64, 22, 64]
        # x ：[B, 22, 64, 64]
        x = input.permute(0,2,1,3).contiguous()

        #########################################
        # Feature normaliization
        # CA ：[B, 1, 64, 64]
        x_ca = self.conv(x)

        # Substraction
        # x_new : [B,22,64,64]
        x_new = x_ca - x 
        # x_new : [B,64,22,64]
        x_new = x_new.permute(0,2,1,3).contiguous()
 
        m_batchsize, C, height, width = x_new.size()

        ############################################
        # Multi-head self-attention Unit
        # x_new : [B,64,22,64]
        x_new = self.query_conv(x_new)
        # proj_query : [B*64,22,64]
        proj_query = x_new.view(-1,height,width)
        # calculate cosine similarity
        # attention : [B*64,22,22]
        y = proj_query / torch.norm(proj_query, 2, 2, keepdim=True)
        attention = torch.bmm(y, y.permute(0, 2, 1))
        # attention : [B,64,22,22]
        attention = attention.view(m_batchsize, C, height,height)



        return attention
