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



class Skeleton(Dataset):

    def __init__(self, dataroot, dataname, phase, input_n=10, output_n=10, action=None):
        
        # load data to form 'all_seqs(data_num, frame_len, fea_dim)'
        if phase == 'train': 
            # self.path_to_data_1 = dataroot +'/' + dataname + '/' + 'train_flip1_c_raw.npy'
            self.path_to_data = dataroot +'/' + dataname + '/' + dataname + phase + 'Data_no_norm_joints.npy'

            all_seqs = np.load(self.path_to_data)
            # all_seqs_1 = np.load(self.path_to_data_1)

            # all_seqs = np.concatenate([all_seqs,all_seqs_1],axis=0)
            print(all_seqs.shape)

        elif phase == 'val': 
            self.path_to_data = dataroot +'/' + dataname + '/' + dataname + phase + 'Data_no_norm_joints.npy'

            all_seqs = np.load(self.path_to_data)
            print(all_seqs.shape)

        else:
            self.path_to_data = dataroot + '/' + dataname + '/' + dataname + '_test_py2.pkl'
            # print(self.path_to_data)
            all_seqs_dict = cPickle.load(open(self.path_to_data,'rb'))
            # print(all_seqs_dict)
            all_seqs = all_seqs_dict[action]
            print(all_seqs.shape)
            


        # reshape all_seqs to (data_num, frame_len, joint_len, dim)
        data_num, frame_len, fea_dim = all_seqs.shape
        # print(all_seqs[0,0,:])
        all_seqs = np.reshape(all_seqs,(data_num,frame_len,-1,3))
        data_num, frame_len, joint_len, dim = all_seqs.shape
            
        # save outputs of raw joint seqs
        self.raw_output_seq = all_seqs[:,input_n:input_n+output_n,:,:]

        # remove some dimensions of raw joint seqs
        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dim_to_use = np.setdiff1d(np.arange(all_seqs.shape[2]), joint_to_ignore)
        
        new_all_seqs = all_seqs[:, :, dim_to_use,:]
        new_all_seqs = np.reshape(new_all_seqs,[-1, 20, 66])

        # 
        self.input_seq = new_all_seqs[:,:input_n,:]
        self.output_seq = new_all_seqs[:,input_n:input_n+output_n,:]

    def __getitem__(self, item):
        return self.input_seq[item], self.output_seq[item],self.raw_output_seq[item]

        # return self.input_seq[item], self.pad_input_seq[item], self.output_seq[item],self.raw_output_seq[item]

    def __len__(self):
        return np.shape(self.input_seq)[0]