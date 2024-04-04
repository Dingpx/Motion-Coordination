import os
import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================

        self.parser.add_argument('--dataroot', type=str, default='.', help='path to dataset')
        self.parser.add_argument('--dataname', type=str, default='h36m', help='type of dataset')
        self.parser.add_argument('--dataname_2', type=str, default='h3.6m_3d', help='type of dataset')

        self.parser.add_argument('--ckpt', type=str, default='checkpoint', help='path to save checkpoint')
        self.parser.add_argument('--test_para', type=str, default='raw', help='paramater of test')
        self.parser.add_argument('--test_type', type=str, default='raw', help='type of test')
        self.parser.add_argument('--device_ids', default=[0,1], type=int, nargs="+", help='device_ids')

        # ===============================================================
        #                     Model options
        # ===============================================================

        self.parser.add_argument('--output_n', default='10', type=int, help='output_n.')
        self.parser.add_argument('--input_n', default='10', type=int, help='input_n.')
        self.parser.add_argument('--joint_len', default='22', type=int, help='joint_len.')
        self.parser.add_argument('--actions', type=str, default='all', help='action type')
        self.parser.add_argument('--hidden_layers', default='1', type=int, help='hidden_layers.')
        self.parser.add_argument('--hidden_units', default='64', type=int, help='hidden_units.')
        self.parser.add_argument('--joint_size', default='3', type=int, help='joint_size.')
        self.parser.add_argument('--num_joints', default='22', type=int, help='num_joints.')
        self.parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')


        # ===============================================================
        #                     Running options
        # ===============================================================

        self.parser.add_argument('--lr', type=float, default=5.0e-4)
        self.parser.add_argument('--epochs', type=int, default=150)
        self.parser.add_argument('--train_batch', type=int, default=16)
        self.parser.add_argument('--val_batch', type=int, default=8)
        self.parser.add_argument('--test_batch', type=int, default=8)
        self.parser.add_argument('--job', type=int, default=2, help='subprocesses to use for data loading')
        self.parser.add_argument('--is_parallel', type=int, default=0, help='flag if paralllel')
        self.parser.add_argument('--is_restart', type=int, default=0, help='flag if is_restart')
        self.parser.add_argument('--deep_supervision', type=int, default=0, help='flag if deep_supervision')


    def _print(self):
        print("\n==================Options===============")
        pprint(vars(self.opt), indent=4)
        print("========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        # do some pre-check
        ckpt = self.opt.ckpt
        # ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self._print()
        return self.opt
