import torch
from torch import nn
import torch.nn.functional as F

from model.traj_block import L_Block1


class UNet1(nn.Module):
    def __init__(self, num_classes, input_channels=3, joint_dim=22):
        super(UNet1,self).__init__()
        nb_filter = [64, 256, 512]

        self.block0_0 = L_Block1(input_channels, nb_filter[0],joint_dim=joint_dim)
        self.block1_0 = L_Block1(nb_filter[0], nb_filter[0],joint_dim=joint_dim)
        self.block2_0 = L_Block1(nb_filter[0], nb_filter[0],joint_dim=joint_dim)
        self.block3_0 = L_Block1(nb_filter[0], nb_filter[0],joint_dim=joint_dim)
        self.block4_0 = L_Block1(nb_filter[0], nb_filter[0],joint_dim=joint_dim)


        self.block2_2 = L_Block1(nb_filter[0]+nb_filter[0], nb_filter[0],joint_dim=joint_dim)
        self.block1_3 = L_Block1(nb_filter[0]+nb_filter[0], nb_filter[0],joint_dim=joint_dim)
        self.block0_4 = L_Block1(nb_filter[0]+nb_filter[0], nb_filter[0],joint_dim=joint_dim)
        self.block8_0 = L_Block1(nb_filter[0]+nb_filter[0], nb_filter[0],joint_dim=joint_dim)

    def forward(self, input,joint_att=0,act=None):

        x0_0,_ = self.block0_0(input)
        x1_0,_ = self.block1_0(x0_0)
        x2_0,_ = self.block2_0(x1_0)
        x3_0,_ = self.block3_0(x2_0)
        x4_0,_ = self.block4_0(x3_0)

        x2_2,_ = self.block2_2(torch.cat([x3_0, x4_0], 1))
        x1_3,_ = self.block1_3(torch.cat([x2_0, x2_2], 1))
        x0_4,_ = self.block0_4(torch.cat([x1_0, x1_3], 1))
        x8_0,_ = self.block8_0(torch.cat([x0_0, x0_4], 1),act=act)

        return x8_0


















