from tools.opts import Options
from tools.h36m import Skeleton

from model.decoder_h36m import Generator

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



class Logger(object):
	def __init__(self, filename=None):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass

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
	file_name = 'train_decoder'

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

	# sys.stdout = Logger(new_ckpt + "/log.txt")

	return new_ckpt

def init_data(opt,acts):
	'''
	load data

	''' 

	test_data = dict()

	for act in acts:
		test_dataset = Skeleton(opt.dataroot, opt.dataname, 'test',output_n = opt.output_n, action = act)

		test_data[act] = DataLoader(
			dataset=test_dataset,
			batch_size=opt.test_batch,
			shuffle=False,
			num_workers=opt.job,
			pin_memory=True)
	print(">>> test data len: {}".format(test_dataset.__len__()*15))

	return test_data

def define_actions():
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    """

    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]

    return actions

def lr_decay(opt,optimizer_d,optimizer_g,lr_now,gamma,epoch):

	if epoch ==2:
		lr = 0.0001
	else:
		lr = lr_now * gamma

	# optimizer_d.param_groups : list(contain one dict)
	# param_group : dict

	for param_group in optimizer_g.param_groups:
		param_group['lr'] = max(lr,0.00002)
	return max(lr,0.00002)

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

	netG = Generator(opt.deep_supervision,opt.input_n)
	is_cuda = torch.cuda.is_available()
	# is_cuda =False

	if opt.is_parallel == 1:
		netG = nn.DataParallel(netG).cuda()

	print(">>> model initinaized")
	print("--------------------------")

	print(">>> inin optimizer")

	if is_cuda:
		netG = netG.cuda()

	one = torch.FloatTensor([1])
	mone = one * -1
	if is_cuda:
		one = one.cuda()
		mone = mone.cuda()

	# params = {"lr": opt.lr}
	# ,"betas": (0.5, 0.999)

	# print(list(sp_layer.parameters()))

	# optimizer_g = torch.optim.Adam(netG.parameters(), lr = opt.lr)


	print(">>> optimizer initinaized")
	print("--------------------------")

	print(">>> test whether ckpt exists")

	last_ckpt_path = new_ckpt + '/' + 'ckpt_' + opt.dataname + '_best.pth.tar'
	print(last_ckpt_path)

	if os.path.exists(last_ckpt_path):

		print(">>> loading ckpt from '{}'".format(last_ckpt_path))
		if is_cuda:
		    ckpt = torch.load(last_ckpt_path)
		else:
		    ckpt = torch.load(last_ckpt_path, map_location='cpu')

		start_epoch = ckpt['epoch']
		err_best = ckpt['best_mjpeg']

		if opt.is_parallel == 1:
			netG.module.load_state_dict(ckpt['G_state_dict'])
			optimizer_g.load_state_dict(ckpt['optimizer_g'])
		else:
			netG.load_state_dict(ckpt['G_state_dict'])
			optimizer_g.load_state_dict(ckpt['optimizer_g'])

		print(">>> ckpt loaded (epoch: {} | best_mjpeg: {} )".format(start_epoch, err_best))
	else:
		print(">>> no existing ckpt")


	return new_ckpt,train_time,val_time,acts,\
			test_data,is_cuda,netG,one,mone,optimizer_g,start_epoch,err_best,all_iters

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
	new_ckpt,train_time,val_time,acts,\
	test_data,is_cuda,netG,one,mone,optimizer_g,start_epoch,err_best,all_iters = inin_all(opt)

	print("===============inin finish==============\n")

	print("===============train start==============")


	# test phase
	test_start = time.time()
	if opt.output_n > 10:
		test_3d_temp = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
	else:
		test_3d_temp = np.array([0.0,0.0,0.0,0.0])

	test_3d_dict ={} 

	for act in acts:
		test_3d = test(test_data[act], netG, act,input_n=opt.input_n, output_n=opt.output_n, is_cuda=is_cuda)
		test_3d_temp += test_3d
		test_3d_dict[act] = test_3d

	for act in acts:

		print('>>> {}: {}    avg:{}'.format(act, test_3d_dict[act], np.mean(test_3d_dict[act])))



	test_3d_temp = test_3d_temp / 15.0
	test_mpjpe = np.mean(test_3d_temp)

	test_epoch_time = time.time() - test_start


	print('>>> all_test_mpjpe: {}   avg:{}'.format(test_3d_temp,test_mpjpe))
	print('>>> epoch_test_time: {}'.format(test_epoch_time))

	# update log file and save checkpoint



def test(test_loader,netG,act,is_cuda=False,output_n=10,input_n=10):

	netG.eval()
	# sp_layer.eval()

	N = 0
	# if output_n == 25:
	# 	eval_frame = [1, 3, 7, 9, 13, 24]
	if output_n == 25:
		eval_frame = [1, 3, 7, 9, 13, 17, 21, 24]

	elif output_n == 10:
		eval_frame = [1, 3, 7, 9]

	t_3d = np.zeros(len(eval_frame))

	for i, (inputs,outputs,raw_outputs) in enumerate(test_loader):

		n = inputs.size(0)
		inputs_length = inputs.size(1)

		if is_cuda:
			outputs = torch.autograd.Variable(outputs.cuda()).float()
			inputs = torch.autograd.Variable(inputs.cuda()).float()
			raw_outputs = torch.autograd.Variable(raw_outputs.cuda()).float()
		else:
			outputs = torch.autograd.Variable(outputs).float()
			inputs = torch.autograd.Variable(inputs).float()
			raw_outputs = torch.autograd.Variable(raw_outputs.float())


		# # inputs = sp_layer(inputs)
		inputs = inputs.contiguous().view(n,inputs_length,-1,3)

		G_fake = netG(inputs,act=act)
		# print(ratio0_0.size())

		if opt.deep_supervision == 1:
			final_pred = G_fake[1]+G_fake[2]
			# final_pred = torch.cat([G_fake[1][:,:5,:,:],G_fake[2][:,5:,:,:]],1)
		else:
			# final_pred = torch.cat([G_fake[0],G_fake[1]],1)
			final_pred = G_fake


		pred_3d = raw_outputs.clone()
		joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
		joint_equal = np.array([13, 19, 22, 13, 27, 30])
		dim_used = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,
							18,19,21,22,25,26,27,29,30])
		pred_3d[:, :, dim_used, :] = final_pred
		pred_3d[:, :, joint_to_ignore, :] = pred_3d[:, :, joint_equal, :]

		for k in np.arange(0, len(eval_frame)):
			j = eval_frame[k]
			t_3d[k] += torch.mean(torch.norm(raw_outputs[:, j, :, :].contiguous().view(-1, 3) 
				- pred_3d[:, j, :, :].contiguous().view(-1, 3), 2, 1)).cpu().data.numpy() * n

		N += n

	return t_3d / N

if __name__ == '__main__':
	opt = Options().parse()
	main(opt)

