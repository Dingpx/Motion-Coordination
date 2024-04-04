from model.non_local_embedded_gaussian import NONLocalBlock2D,NONLocalBlock1D
from model.se_block import Se_layer
from model.traj_block import L_Block
from model.unet_block import UNet1
from model.point_fea_block import Multi_T2_abalation_test02

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding, init
from mmcv.cnn import constant_init, kaiming_init
import pdb
import math

class Generator(nn.Module):

    def __init__(self,deep_supervision_flag,ch_in,ch_out):
        super(Generator,self).__init__()

        in_channel=3
        out_channel=32
        num_joint=22
        num_person=2
        window_size=64
        num_class = 60
        self.time = ch_out
        joint_dim =25

        self.encoder_d0 = Multi_T2_abalation_test02(in_channel,out_channel)
        self.encoder_d1 = Multi_T2_abalation_test02(in_channel,out_channel)

        self.encoder_t = UNet1(num_classes=10,input_channels=19,joint_dim=joint_dim)
        self.decoder_t = nn.Conv2d(64, self.time, kernel_size=1)
        self.decoder_d = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, inputs):

        last_input = torch.unsqueeze(inputs[:,9,:,:],dim=1).repeat(1,self.time,1,1)
        motion = inputs[:,1::,:,:]-inputs[:,0:-1,:,:]

        x = inputs.permute(0,3,2,1).contiguous()
        motion = motion.permute(0,3,2,1).contiguous()

        # point fea
        out_0_0 = self.encoder_d0(x).permute(0,3,2,1)
        out_0_1 = self.encoder_d1(motion).permute(0,3,2,1)
        out_1 = torch.cat((out_0_0,out_0_1),dim=1)

        # encode traj
        out_2 = self.encoder_t(out_1,joint_att=0)

        # decode traj and dim
        output = self.decoder_t(out_2)
        
        # output = out_2
        output = self.decoder_d(output.permute(0,3,2,1)).permute(0,3,2,1)

        outputs = output + last_input
        # outputs = output

        return outputs
