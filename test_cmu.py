from tools.opts import Options
from tools.cmu_test import Skeleton

from model.decoder_cmu import Generator

import _pickle as cPickle
import math
import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from progress.bar import Bar


from torch.autograd import Variable
import numpy as np
import pandas as pd
import pdb
import shutil
import random
import sys
import pickle

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_ckpt(opt):
	'''
	inialize new ckpt route

	'''
	# file_name = 'train' + __file__.split('/')[0][4:-3]

	# for long-term prediciton
	file_name = 'train_decoder_cmu_long'

	test_file_name = 'test' + __file__.split('/')[0][4:-3]

	if opt.is_parallel == 1:
		new_ckpt = opt.ckpt + '/' + file_name+ '/' + opt.dataname + '/'+ opt.test_type + '/' + opt.test_para + '/' + 'parallel'
	else:
		new_ckpt = opt.ckpt + '/' + file_name+ '/' + opt.dataname + '/'+ opt.test_type + '/' + opt.test_para + '/' + 'single'

	if opt.is_restart == 1:
		if os.path.exists(new_ckpt):
			shutil.rmtree(new_ckpt)
		else:
			pass

	if os.path.exists(new_ckpt):
		pass
	else:
		os.makedirs(new_ckpt)

	return new_ckpt

def init_data(opt,acts):
	'''
	load data

	''' 

	test_data = dict()
	for act in acts:
		test_dataset = Skeleton(actions= act,input_n=opt.input_n, output_n=opt.output_n, split=1)

		test_data[act] = DataLoader(
			dataset=test_dataset,
			batch_size=opt.test_batch,
			shuffle=False,
			num_workers=opt.job,
			pin_memory=True)
	print(">>> test data len: {}".format(test_dataset.__len__()*8))


	# with open('test_256_long.pkl', 'wb') as f:
	# 	pickle.dump(test_dict, f, pickle.HIGHEST_PROTOCOL)

	return test_data

def define_actions():
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    """

    actions = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
               "washwindow"]

    return actions

def inin_all(opt):

	print(">>> init parameter")

	new_ckpt = init_ckpt(opt)

	start_epoch = 1
	all_iters = 0
	err_best = 100000000
	# k = 0.0

	train_time = AverageMeter()
	val_time = AverageMeter()
	
	print(">>> parameter initinaized")
	print("--------------------------")

	print(">>> loading data")

	acts = define_actions()
	test_data = init_data(opt,acts)

	print(">>> data loaded !")
	print("--------------------------")

	print(">>> inin model")

	netG = Generator(opt.deep_supervision,opt.input_n,opt.output_n)
	is_cuda = torch.cuda.is_available()
	# is_cuda =False

	if opt.is_parallel == 1:
		netG = nn.DataParallel(netG).cuda()

	print(">>> model initinaized")
	print("--------------------------")

	print(">>> inin optimizer")

	if is_cuda:
		netG = netG.cuda()

	print(">>> optimizer initinaized")
	print("--------------------------")

	print(">>> test whether ckpt exists")

	best_ckpt_path = new_ckpt + '/' + 'ckpt_' + opt.dataname + '_best.pth.tar'
	print(best_ckpt_path)

	if os.path.exists(best_ckpt_path):

		print(">>> loading ckpt from '{}'".format(best_ckpt_path))
		if is_cuda:
		    ckpt = torch.load(best_ckpt_path)
		else:
		    ckpt = torch.load(best_ckpt_path, map_location='cpu')

		epoch = ckpt['epoch']
		err_best = ckpt['best_mjpeg']

		if opt.is_parallel == 1:
			netG.module.load_state_dict(ckpt['G_state_dict'])
		else:
			netG.load_state_dict(ckpt['G_state_dict'])

		print(">>> ckpt loaded (epoch: {} | best_mjpeg: {} )".format(epoch, err_best))
	else:
		print(">>> no existing ckpt")


	return new_ckpt,acts,test_data,is_cuda,netG,epoch,err_best,all_iters

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def main(opt):

	torch.backends.cudnn.deterministic = True

	LAMBDA = 10
	GLOBAL_SEED = 2020
	set_seed(GLOBAL_SEED)

	lr_now = opt.lr
	os.environ["CUDA_VISIBLE_DEVICES"] = '1'

	print("================inin start==============")
	new_ckpt,acts,\
	test_data,is_cuda,netG,epoch,err_best,all_iters = inin_all(opt)

	print("===============inin finish==============\n")

	print("===============train start==============")


	# test phase
	test_start = time.time()
	if opt.output_n > 10:
		test_3d_temp = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
	else:
		test_3d_temp = np.array([0.0,0.0,0.0,0.0])

	test_3d_dict ={} 

	for act in acts:
		test_3d = test(test_data[act], netG, act,input_n=opt.input_n, output_n=opt.output_n, is_cuda=is_cuda)
		test_3d_temp += test_3d
		test_3d_dict[act] = test_3d

	for act in acts:

		print('>>> {}: {}    avg:{}'.format(act, test_3d_dict[act], np.mean(test_3d_dict[act])))

	test_3d_temp = test_3d_temp / 8.0
	test_mpjpe = np.mean(test_3d_temp)

	test_epoch_time = time.time() - test_start

	print('>>> all_test_mpjpe: {}   avg:{}'.format(test_3d_temp,test_mpjpe))
	print('>>> epoch_test_time: {}'.format(test_epoch_time))

def test(test_loader,netG,act,is_cuda=False,output_n=10,input_n=10):

	netG.eval()

	N = 0
	if output_n == 25:
		eval_frame = [1, 3, 7, 9, 13, 24]

	elif output_n == 10:
		eval_frame = [1, 3, 7, 9]

	t_3d = np.zeros(len(eval_frame))

	for i, (inputs,outputs,raw_outputs,raw_inputs) in enumerate(test_loader):

		n = inputs.size(0)
		inputs_length = inputs.size(1)
		outputs_length = outputs.size(1)

		if is_cuda:
			outputs = torch.autograd.Variable(outputs.cuda()).float()
			inputs = torch.autograd.Variable(inputs.cuda()).float()
			raw_outputs = torch.autograd.Variable(raw_outputs.cuda()).float()
			raw_inputs = torch.autograd.Variable(raw_inputs.cuda()).float()


		else:
			outputs = torch.autograd.Variable(outputs).float()
			inputs = torch.autograd.Variable(inputs).float()
			raw_outputs = torch.autograd.Variable(raw_outputs.float())
			raw_inputs = torch.autograd.Variable(raw_inputs.float())



		inputs = inputs.contiguous().view(n,inputs_length,-1,3)
		outputs = outputs.contiguous().view(n,outputs_length,-1,3)


		raw_inputs1 = raw_inputs.contiguous().view(n,inputs_length,-1,3)
		np.save('/home/dpx/Abalation/vis/cmu/long_term/input/input_'+act,raw_inputs1.detach().cpu().numpy())
		raw_outputs1 = raw_outputs.contiguous().view(n,outputs_length,-1,3)
		np.save('/home/dpx/Abalation/vis/cmu/long_term/gt/gt_'+act,raw_outputs1.detach().cpu().numpy())

		G_fake = netG(inputs)


		if opt.deep_supervision == 1:
			final_ptd = G_fake[1]+G_fake[2]
			# final_pred = torch.cat([G_fake[1][:,:5,:,:],G_fake[2][:,5:,:,:]],1)
		else:
			# final_pred = torch.cat([G_fake[0],G_fake[1]],1)
			final_pred = G_fake

		pred_3d = raw_outputs.clone()
		joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
		joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
		dim_used = np.array([3,4,5,6,9,10,11,12,14,15,17,18,19,21,22,23,25,26,28,30,31,32,34,35,37])

		print(pred_3d.size())
		print(final_pred.size())
		pred_3d[:, :, dim_used, :] = final_pred
		pred_3d[:, :, joint_to_ignore, :] = pred_3d[:, :, joint_equal, :]

		np.save('/home/dpx/Abalation/vis/cmu/long_term/my_pred/pred_'+act,pred_3d.detach().cpu().numpy())

		# bone_loss = cal_bone_loss_32(raw_outputs,pred_3d)

		# print(bone_loss)

		for k in np.arange(0, len(eval_frame)):
			j = eval_frame[k]
			t_3d[k] += torch.mean(torch.norm(raw_outputs[:, j, :, :].contiguous().view(-1, 3) 
				- pred_3d[:, j, :, :].contiguous().view(-1, 3), 2, 1)).cpu().data.numpy() * n

		N += n

	return t_3d / N


if __name__ == '__main__':
	opt = Options().parse()
	main(opt)

