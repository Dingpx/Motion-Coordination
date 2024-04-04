from tools.opts import Options
from tools.h36m_long import Skeleton
from model.decoder_h36m_long import Generator

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

def get_ratio2():
	# index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
	# y1 = []
	# y1_norm = []
	# for i in index:
	# 	y = pow(math.pi,-0.3*i)
	# 	y1.append(y)

	# y1_sum = sum(y1)

	# for i in y1:
	# 	y1_norm.append(i/y1_sum)

	# return y1_norm


	y1 = torch.cuda.FloatTensor([0.4, 0.35, 0.3, 0.28, 0.26, 0.24, 0.21, 0.20, 0.19,
    0.19, 0.18, 0.18, 0.17, 0.17, 0.17, 0.17, 0.17, 0.16,
    0.16, 0.16, 0.16, 0.15, 0.15, 0.15, 0.15])

	y1 = torch.unsqueeze(y1,dim=1)

	return y1

def get_martrix():

    a = torch.zeros((32,22)).cuda()
    a[1,0] = 1
    a[2,0] = -1
    a[2,1] = 1
    a[3,1] = -1
    a[3,2] = 1
    a[4,2] = -1
    a[4,3] = 1
    a[5,3] = -1
    a[6,4] = 1
    a[7,4] = -1
    a[7,5] = 1
    a[8,5] = -1
    a[8,6] = 1
    a[9,6] = -1
    a[9,7] = 1
    a[10,7] = -1
    a[13,8] = 1
    a[25,8] = -1
    a[25,9] = 1
    a[26,9] = -1
    a[26,10] = 1
    a[27,10] = -1
    a[27,11] = 1
    a[28,11] = -1
    a[27,12] = 1
    a[30,12] = -1
    a[13,13] = 1
    a[17,13] = -1
    a[17,14] = 1
    a[18,14] = -1
    a[18,15] = 1
    a[19,15] = -1
    a[19,16] = 1
    a[20,16] = -1
    a[19,17] = 1
    a[22,17] = -1
    a[11,18] = 1
    a[12,18] = -1
    a[12,19] = 1
    a[13,19] = -1
    a[13,20] = 1
    a[14,20] = -1
    a[14,21] = 1
    a[15,21] = -1
    return a

def cal_bone_loss(outputs,G_fake):

	raw_bone_length = bone_loss(outputs)
	fake_bone_length = bone_loss(G_fake)

	diff = (fake_bone_length - raw_bone_length)**2 # 平方误差 torch.abs()
	loss = torch.mean(diff) #平均每个bone的平方误差

	return loss

def cal_bone_loss_32(outputs,G_fake,ratio=None):

	raw_bone_length = bone_loss_32(outputs)
	fake_bone_length = bone_loss_32(G_fake)

	# diff = torch.abs(fake_bone_length - raw_bone_length) # 平方误差 torch.abs()
	diff = (fake_bone_length - raw_bone_length)**2 # 平方误差 torch.abs()

	diff = diff.view(-1)

	if ratio is not None:

		diff = diff * ratio
		loss = torch.mean(diff) #平均每个bone的平方误差
	else:
		loss = torch.mean(diff) #平均每个bone的平方误差

	return loss

def bone_loss(x):
    # KCS 
    batch_num = x.size()[0]
    frame_num = x.size()[1]
    joint_num = x.size()[2]
    bone_length = torch.FloatTensor(batch_num*frame_num,joint_num-1).cuda()

    # inter_loss = 0.

    Ct = torch.cuda.FloatTensor([
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]])

    x_ = x.transpose(2, 3) # b, t, 3, 22
    x_ = torch.matmul(x_, Ct)  # b, t, 3, 21
    bone_length = torch.norm(x_, 2, 2) # b, t, 21

    return bone_length

def bone_loss_32(x):
    # KCS 
    batch_num = x.size()[0]
    frame_num = x.size()[1]
    joint_num = x.size()[2]

    Ct = get_martrix()

    x_ = x.transpose(2, 3) # b, t, 3, 22
    x_ = torch.matmul(x_, Ct)  # b, t, 3, 21
    bone_length = torch.norm(x_, 2, 2) # b, t, 21

    return bone_length

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
	file_name = __file__.split('/')[0][:-3]

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

	sys.stdout = Logger(new_ckpt + "/log.txt")

	return new_ckpt

def init_data(opt,acts):
	'''
	load data

	''' 

	train_dataset = Skeleton(opt.dataroot, opt.dataname, opt.dataname_2,'train',output_n = opt.output_n)
	train_loader = DataLoader(
		dataset=train_dataset,
		batch_size=opt.train_batch,
		shuffle=True,
		num_workers=opt.job,
		pin_memory=True)
	print(">>> train data len: {}".format(train_dataset.__len__()))

	test_data = dict()
	for act in acts:
		test_dataset = Skeleton(opt.dataroot, opt.dataname, opt.dataname_2,'test',output_n = opt.output_n, action = act)

		test_data[act] = DataLoader(
			dataset=test_dataset,
			batch_size=opt.test_batch,
			shuffle=False,
			num_workers=opt.job,
			pin_memory=True)
	print(">>> test data len: {}".format(test_dataset.__len__()*15))

	return train_loader, test_data

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

def lr_decay(opt,optimizer_g,lr_now,gamma,epoch):

	if epoch==4:
		lr = 0.0001
	else:
		lr = lr_now * gamma
	# lr = lr_now * gamma


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
	train_loader, test_data = init_data(opt,acts)

	print(">>> data loaded !")
	print("--------------------------")

	print(">>> inin model")

	if opt.is_parallel == 1:
		os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

	netG = Generator(opt.deep_supervision,opt.input_n)
	is_cuda = torch.cuda.is_available()

	if is_cuda:
		netG = netG.cuda()

	if opt.is_parallel == 1:
		netG = nn.DataParallel(netG)

	print(">>> model initinaized")
	print("--------------------------")

	print(">>> inin optimizer")


	one = torch.FloatTensor([1])
	mone = one * -1
	if is_cuda:
		one = one.cuda()
		mone = mone.cuda()

	optimizer_g = torch.optim.Adam(netG.parameters(), lr = opt.lr)

	print(">>> optimizer initinaized")
	print("--------------------------")

	print(">>> test whether ckpt exists")

	last_ckpt_path = new_ckpt + '/' + 'ckpt_' + opt.dataname + '_last.pth.tar'

	if os.path.exists(last_ckpt_path):

		print(">>> loading ckpt from '{}'".format(last_ckpt_path))
		if is_cuda:
		    ckpt = torch.load(last_ckpt_path)
		else:
		    ckpt = torch.load(last_ckpt_path, map_location='cpu')

		start_epoch = ckpt['epoch']
		err_best = ckpt['best_mjpeg']

		if opt.is_parallel == 1:
			# netD.module.load_state_dict(ckpt['D_state_dict'])
			netG.module.load_state_dict(ckpt['G_state_dict'])
			optimizer_g.load_state_dict(ckpt['optimizer_g'])
		else:
			# netD.load_state_dict(ckpt['D_state_dict'])
			netG.load_state_dict(ckpt['G_state_dict'])
			optimizer_g.load_state_dict(ckpt['optimizer_g'])

		print(">>> ckpt loaded (epoch: {} | best_mjpeg: {} )".format(start_epoch, err_best))
	else:
		print(">>> no existing ckpt")


	return new_ckpt,train_time,val_time,acts,train_loader,\
			test_data,is_cuda,netG,one,mone,optimizer_g,start_epoch,err_best

def cal_mpjpe_torch(raw_outputs, pre_seq):

	pred_3d = pre_seq.contiguous()
	targ_3d = raw_outputs.contiguous()
	diff = torch.norm(pre_seq - raw_outputs, 2, 3).view(-1)
	mpjpe = torch.mean(diff)

	

	return mpjpe

def save_ckpt(state, ckpt_path, is_best=True, file_name=['ckpt_best.pth.tar', 'ckpt_last.pth.tar'],is_load = False,epoch=None):
	
	file_path = os.path.join(ckpt_path, file_name[1])
	torch.save(state, file_path)
	if is_best:
		file_path = os.path.join(ckpt_path, file_name[0])
		torch.save(state, file_path)
	if epoch == 50:
		file_path = os.path.join(ckpt_path, 'ckpt_50.pth.tar')
		torch.save(state, file_path)

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
	new_ckpt,train_time,val_time,acts,train_loader,\
	test_data,is_cuda,netG,one,mone,optimizer_g,start_epoch,err_best = inin_all(opt)

	print("===============inin finish==============\n")

	print("===============train start==============")
	for epoch in range(start_epoch, opt.epochs+1):

		# weight decay by lr_now = 0.96 * lr_last

		if epoch > 1 :
		# if epoch >1 :
			lr_now = lr_decay(opt,optimizer_g,lr_now,0.96,epoch)

		print('--------------------------')
		print('>>> epoch: {} | lr: {:.8f}'.format(epoch, lr_now))

		# add epoch into pd
		ret_log = np.array([epoch])
		head = np.array(['epoch'])

		# train phase 
		train_start = time.time()

		mjpeg_l,bone_l,total_l= train(opt,train_loader,netG, 
			optimizer_g,epoch,one,mone,LAMBDA,is_cuda=is_cuda,input_n=opt.input_n,output_n=opt.output_n,
			test_data=test_data,acts=acts,err_best=err_best,new_ckpt=new_ckpt,
			start_epoch = start_epoch)

		train_epoch_time = time.time() - train_start
		train_time.update(train_epoch_time)

		print('>>> epoch_train_time: {:.2f}'.format(train_epoch_time))
		print('>>> mjpeg_l: {:.2f}'.format(mjpeg_l))
		print('>>> bone_l: {:.2f}'.format(bone_l))
		print('>>> total_l: {:.2f}'.format(total_l))

		print(' 							')


		ret_log = np.append(ret_log,[mjpeg_l,bone_l,total_l,train_epoch_time])
		head = np.append(head, ['mjpeg_l','bone_l','total_l','train_epoch_time'])

		# test phase
		test_start = time.time()
		if opt.output_n > 10:
			test_3d_temp = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
		else:
			test_3d_temp = np.array([0.0,0.0,0.0,0.0])

		test_3d_dict ={}


		for act in acts:
			test_3d = test(test_data[act], netG, act,input_n=opt.input_n, output_n=opt.output_n, is_cuda=is_cuda)
			# print(test_3d)
			# ret_log = np.append(ret_log, test_3d)
			# head = np.append(head,
			# 		['test_' + act + '_3d80', 'test_' + act + '_3d160', 'test_' + act + '_3d320', 'test_' + act + '_3d400'])
			# if opt.output_n > 10:
			# 	head = np.append(head, ['test_' + act + '_3d560', 'test_' + act + '_3d1000'])
			test_3d_temp += test_3d
			test_3d_dict[act] = test_3d



		for act in acts:

			print('>>> {}: {}    avg:{}'.format(act, test_3d_dict[act], np.mean(test_3d_dict[act])))

			ret_log = np.append(ret_log, test_3d_dict[act])
			head = np.append(head,
				['test_' + act + '_3d80', 'test_' + act + '_3d160', 'test_' + act + '_3d320', 'test_' + act + '_3d400'])
			if opt.output_n > 10:
				head = np.append(head, ['test_' + act + '_3d560', 'test_' + act + '_3d1000'])

		test_3d_temp = test_3d_temp / 15.0
		test_mpjpe = np.mean(test_3d_temp)

		test_epoch_time = time.time() - test_start
		ret_log = np.append(ret_log, [test_epoch_time])
		head = np.append(head, [ 'test_epoch_time'])

		print('>>> all_test_mpjpe: {}   avg:{}'.format(test_3d_temp,test_mpjpe))
		print('>>> epoch_test_time: {}'.format(test_epoch_time))

		ret_log = np.append(ret_log, test_3d_temp)
		head = np.append(head, ['test_all_3d80', 'test_all_3d160', 'test_all_3d320', 'test_all_3d400'])
		if opt.output_n > 10:
			head = np.append(head, ['test_all_3d560', 'test_all_3d1000'])

		# update log file and save checkpoint
		df = pd.DataFrame(np.expand_dims(ret_log, axis=0))

		if epoch == start_epoch:
			df.to_csv(new_ckpt + '/' + opt.dataname + '.csv', header=head, index=False)
		else:
			with open(new_ckpt + '/' + opt.dataname + '.csv', 'a') as f:
				df.to_csv(f, header=False, index=False)

		# 存储ckpt
		if not np.isnan(test_mpjpe):
			is_best = test_mpjpe < err_best
			err_best = min(test_mpjpe, err_best)
		else:
			is_best = False

		file_name = ['ckpt_' + opt.dataname + '_best.pth.tar', 'ckpt_' + opt.dataname + '_last.pth.tar']
		if opt.is_parallel == 1:
			save_ckpt({'epoch': epoch,
						'best_mjpeg': err_best,
						'G_state_dict': netG.module.state_dict(),
						'optimizer_g': optimizer_g.state_dict(),

						},
						ckpt_path=new_ckpt + '/',
						is_best=is_best,
						file_name=file_name,
						epoch = epoch)
		else:
			save_ckpt({'epoch': epoch,
						'best_mjpeg': err_best,
						'G_state_dict': netG.state_dict(),
						'optimizer_g': optimizer_g.state_dict(),
						},
						ckpt_path=new_ckpt + '/',
						is_best=is_best,
						file_name=file_name,
						epoch = epoch)

def train(opt,train_loader, netG, optimizer_g, epoch, one, mone, LAMBDA, is_cuda=False,
	input_n= None,output_n =None,test_data=None,acts=None,err_best=None,new_ckpt=None,start_epoch=None,k=None):

	mpjpe_l = AverageMeter()
	bone_l = AverageMeter()
	total_l = AverageMeter()

	st = time.time()
	bar = Bar('>>>', fill='>', max=len(train_loader))

	netG.train()

	for iter, (inputs,outputs,raw_outputs) in enumerate(train_loader):

		n = inputs.size(0)
		inputs_length = inputs.size(1)
		outputs_length = outputs.size(1)
		if is_cuda:
			outputs = torch.autograd.Variable(outputs.cuda()).float()
			inputs = torch.autograd.Variable(inputs.cuda()).float()
			raw_outputs = torch.autograd.Variable(raw_outputs.cuda()).float()

		else:
			outputs = torch.autograd.Variable(outputs).float()
			inputs = torch.autograd.Variable(inputs).float()
			raw_outputs = torch.autograd.Variable(raw_outputs.float())


		outputs = outputs.contiguous().view(n,outputs_length,-1,3)
		inputs = inputs.contiguous().view(n,inputs_length,-1,3)

		for p in netG.parameters(): 
			p.requires_grad = True

		bt = time.time()

		G_fake = netG(inputs)


		if opt.deep_supervision == 1:
			final_ptd = G_fake[1]+G_fake[2]
			mpjpe = cal_mpjpe_torch(outputs, G_fake) 
		else:
			final_pred = G_fake
			mpjpe = cal_mpjpe_torch(outputs, G_fake) 

		bone_loss = cal_bone_loss(outputs,G_fake)

		if epoch>4:

			total_loss = mpjpe

			optimizer_g.zero_grad()
			total_loss.backward()
			optimizer_g.step()

			mpjpe_l.update(mpjpe.item(),n)
			bone_l.update(bone_loss.item(),n)
			total_l.update(total_loss.item(),n)

		else:

			total_loss = mpjpe

			optimizer_g.zero_grad()
			mpjpe.backward()
			optimizer_g.step()

			mpjpe_l.update(mpjpe.item(),n)
			bone_l.update(bone_loss.item(),n)

		bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(iter+1, len(train_loader), time.time() - bt,time.time() - st)
		bar.next()
	bar.finish()

	return mpjpe_l.avg,bone_l.avg,total_l.avg

def test(test_loader,netG,act,is_cuda=False,output_n=25,input_n=10):

	netG.eval()

	N = 0
	if output_n == 25:
		eval_frame = [1, 3, 7, 9, 13, 24]

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


		inputs = inputs.contiguous().view(n,inputs_length,-1,3)

		G_fake = netG(inputs)

		if opt.deep_supervision == 1:
			final_pred = G_fake[1]+G_fake[2]
			# final_pred = torch.cat([G_fake[1][:,:5,:,:],G_fake[2][:,5:,:,:]],1)
		else:
			# final_pred = torch.cat([G_fake[0],G_fake[1]],1)
			final_pred = G_fake



		# G_fake = G_fake.contiguous().view(n,inputs_length,-1)
		# G_fake = G_fake.contiguous().view(n,inputs_length,-1,3)


		pred_3d = raw_outputs.clone()
		joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
		joint_equal = np.array([13, 19, 22, 13, 27, 30])
		dim_used = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,
							18,19,21,22,25,26,27,29,30])
		pred_3d[:, :, dim_used, :] = final_pred
		pred_3d[:, :, joint_to_ignore, :] = pred_3d[:, :, joint_equal, :]

		bone_loss = cal_bone_loss_32(raw_outputs,pred_3d)

		print(bone_loss)

		for k in np.arange(0, len(eval_frame)):
			j = eval_frame[k]
			t_3d[k] += torch.mean(torch.norm(raw_outputs[:, j, :, :].contiguous().view(-1, 3) 
				- pred_3d[:, j, :, :].contiguous().view(-1, 3), 2, 1)).cpu().data.numpy() * n

		N += n

	return t_3d / N

if __name__ == '__main__':
	opt = Options().parse()
	main(opt)

