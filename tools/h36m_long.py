from torch.utils.data import Dataset
import numpy as np
import os
import _pickle as cPickle
from util import data_utils



H36M_SKELETON = [
    [(-1, 0, "RightLeg"), (-1, 4, "LeftLeg"), (-1, 8, "Spine1")],
    [(0, 1, "RightFoot"), (4, 5, "LeftFoot"), (8, 17, "RightArm"), (8, 12, "LeftArm"), (8, 9, "Neck")],
    [(1, 2, "RightToeBase"), (5, 6, "LeftToeBase"), (17, 18, "RightForeArm"), (12, 13, "LeftForeArm"), (9, 10, "Head")],
    [(2, 3, "RightToeBase_Site"), (6, 7, "LeftToeBase_Site"),(18, 19, "RightHand"), (13, 14, "LeftHand"), (10, 11, "Head_Site")],
    [(19, 20, "RightHand_Site"), (14, 15, "LeftHand_Site")],
    [(20, 21, "R_Wrist_End"), (15, 16, "L_Wrist_End")]
]

def Skeleton1(dataroot, dataname, dataname_2, phase, input_n=10, output_n=25, action=None):
    path_to_data = './h36m/h3.6m/dataset/'

    all_seqs, dim_ignore1, dim_use1 = data_utils.load_data_3d_256(path_to_dataset=path_to_data, subjects=[5], actions=[action],sample_rate=2, seq_len=input_n+output_n)
    print(all_seqs.shape)

    return all_seqs

class Skeleton(Dataset):

    def __init__(self, dataroot, dataname, dataname_2, phase, input_n=10, output_n=25, action=None):

        # load data to form 'all_seqs(data_num, frame_len, fea_dim)'
        if phase == 'train' or phase == 'val': 

            self.path_to_data = dataroot +'/' + dataname + '/' + dataname_2 + '/' + 'h36m_xyz_' + phase + '_no_noma_all_joint.npy'
            all_seqs = np.load(self.path_to_data)
            # print(self.path_to_data)

        else:

            # self.path_to_data = './test_256_long.pkl'
            # all_seqs_dict = cPickle.load(open(self.path_to_data,'rb'))
            # all_seqs = all_seqs_dict[action]
            path_to_data = '../LTD/h3.6m/dataset'

            all_seqs, dim_ignore1, dim_use1 = data_utils.load_data_3d(path_to_dataset=path_to_data, subjects=[5], actions=[action],
                                                        sample_rate=2, seq_len=input_n+output_n)
            print(all_seqs.shape)

        data_num, frame_len, fea_dim = all_seqs.shape
        all_seqs = np.reshape(all_seqs,(data_num,frame_len,-1,3))
        data_num, frame_len, joint_len, dim = all_seqs.shape
            
        # save outputs of raw joint seqs
        self.raw_output_seq = all_seqs[:,input_n:input_n+output_n,:,:]

        # remove some dimensions of raw joint seqs
        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dim_to_use = np.setdiff1d(np.arange(all_seqs.shape[2]), joint_to_ignore)
        new_all_seqs = all_seqs[:, :, dim_to_use,:]
        new_all_seqs = np.reshape(new_all_seqs,[-1, 35, 66])

        self.input_seq = new_all_seqs[:,:input_n,:]
        self.output_seq = new_all_seqs[:,input_n:input_n+output_n,:]


    def __getitem__(self, item):
        return self.input_seq[item], self.output_seq[item],self.raw_output_seq[item]

        # return self.input_seq[item], self.pad_input_seq[item], self.output_seq[item],self.raw_output_seq[item]

    def __len__(self):
        return np.shape(self.input_seq)[0]