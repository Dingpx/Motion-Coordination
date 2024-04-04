from torch.utils.data import Dataset
import torch
import numpy as np
from util import data_utils


class Skeleton(Dataset):

    def __init__(self, actions='all',input_n=10, output_n=10, split=0, data_mean=0, data_std=0, dim_used=0,
                 dct_n=15):

        path_to_data = './cmu_mocap'
        self.split = split
        actions = data_utils.define_actions_cmu(actions)
        print(actions)
        # actions = ['walking']
        if split == 0:
            path_to_data = path_to_data + '/train'
            is_test = False
        else:
            path_to_data = path_to_data + '/test'
            is_test = True
        all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d(path_to_data, actions,
                                                                                         input_n, output_n,
                                                                                         data_std=data_std,
                                                                                         data_mean=data_mean,
                                                                                         is_test=is_test)
        if not is_test:
            dim_used = dim_use

        self.all_seqs = all_seqs
        self.dim_used = dim_used

        # print('debugging')
        # if split == 0:
        #     all_seqs = all_seqs[:50,:,:]
        #     data_num = 50
        # else:
        #     all_seqs = all_seqs[:,:,:]


        data_num, frame_len, fea_dim = all_seqs.shape
        # print(all_seqs[0,0,:])
        all_seqs = np.reshape(all_seqs,(data_num,frame_len,-1,3))
        data_num, frame_len, joint_len, dim = all_seqs.shape


        # save outputs of raw joint seqs
        self.raw_output_seq = all_seqs[:,input_n:input_n+output_n,:,:]
        # self.raw_input_seq = all_seqs[:,:input_n,:,:]
        


        # remove some dimensions of raw joint seqs
        joint_to_ignore = np.array([0, 1, 2, 7, 8, 13, 16, 20, 29, 24, 27, 33, 36])
        dim_to_use = np.setdiff1d(np.arange(all_seqs.shape[2]), joint_to_ignore)
        
        new_all_seqs = all_seqs[:, :, dim_to_use,:]
        new_all_seqs = np.reshape(new_all_seqs,[-1, input_n+output_n, 75])

        # 
        self.input_seq = new_all_seqs[:,:input_n,:]
        self.output_seq = new_all_seqs[:,input_n:input_n+output_n,:]
        
        print(self.output_seq.shape)
        print(self.input_seq.shape)

        print(all_seqs.shape)




    def __len__(self):
        return np.shape(self.input_seq)[0]

    def __getitem__(self, item):
        return self.input_seq[item], self.output_seq[item],self.raw_output_seq[item]
