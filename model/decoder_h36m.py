from model.unet_block import UNet1
from model.point_fea_block import Multi_T2_abalation_test02

import torch
from torch import nn
import torch.nn.functional as F
import pdb
import math


class Generator(nn.Module):

    def __init__(self,deep_supervision_flag,ch_in):
        super(Generator,self).__init__()

        in_channel=3
        out_channel=32
        num_joint=22
        num_person=2
        window_size=64
        num_class = 60
        time = 10

        self.encoder_d0 = Multi_T2_abalation_test02(in_channel,out_channel)
        self.encoder_d1 = Multi_T2_abalation_test02(in_channel,out_channel)

        self.encoder_t = UNet1(num_classes=10,input_channels=19,joint_dim=22)

        self.decoder_t = nn.Conv2d(64, 10, kernel_size=1)
        self.decoder_d = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, inputs, act=None):

        motion = inputs[:,1::,:,:]-inputs[:,0:-1,:,:]

        x = inputs.permute(0,3,2,1).contiguous()
        motion = motion.permute(0,3,2,1).contiguous()

        # point fea
        out_0_0 = self.encoder_d0(x).permute(0,3,2,1)
        out_0_1 = self.encoder_d1(motion).permute(0,3,2,1)
        out_1 = torch.cat((out_0_0,out_0_1),dim=1)

        # encode traj
        out_2 = self.encoder_t(out_1,joint_att=0,act=act)

        # decode traj and dim
        output = self.decoder_t(out_2)
        
        # output = out_2
        output = self.decoder_d(output.permute(0,3,2,1)).permute(0,3,2,1)

        # outputs = output + last_input
        outputs = output + inputs

        return outputs
