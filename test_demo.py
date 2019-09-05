import numpy as np
from scipy import stats
import scipy.io as sio
import os
import torch
from torch import nn
from dataloader import Data_Loader
import util
import torch.utils.data
from torchfoldext import FoldExt
import sys
import json
import partnet as partnet_model

with open('./part_color_mapping.json', 'r') as f:
	color = json.load(f)

for c in color:
	c[0] = int(c[0]*255)
	c[1] = int(c[1]*255)
	c[2] = int(c[2]*255)

def writeply(savedir, data, label):
	path = os.path.dirname(savedir)
	if not os.path.exists(path):
		os.makedirs(path)
	if data.size(0) == 0:
		n_vertex = 0
	else:
		n_vertex = data.size(1)
	with open(savedir, 'w') as f:
		f.write('ply\n')
		f.write('format ascii 1.0\n')
		f.write('comment 111231\n')
		f.write('element vertex %d\n' % n_vertex)
		f.write('property float x\n')
		f.write('property float y\n')
		f.write('property float z\n')
		f.write('property float nx\n')
		f.write('property float ny\n')
		f.write('property float nz\n')
		f.write('property uchar red\n')
		f.write('property uchar green\n')
		f.write('property uchar blue\n')
		f.write('property uchar label\n')
		f.write('end_header\n')
		for j in range(n_vertex):
			f.write('%g %g %g %g %g %g %d %d %d %d\n' % (*data[0, j], *color[label[j]], label[j]))

def normalize_shape(shape):
	shape_min, _ = torch.min(shape[:,:,:3], 1)
	shape_max, _ = torch.max(shape[:,:,:3], 1)
	x_length = shape_max[0, 0] - shape_min[0, 0]
	y_length = shape_max[0, 1] - shape_min[0, 1]
	z_length = shape_max[0, 2] - shape_min[0, 2]
	length = shape_max - shape_min
	max_length = torch.max(length)
	scale = 1/max_length
	center = shape_min + length/2
	print(center[0, 0],center[0, 1],center[0, 2], scale)
	for i in range(2048):
		shape[0, i, 0] = (shape[0, i, 0] - center[0, 0])* scale
		shape[0, i, 1] = (shape[0, i, 1] - center[0, 1])* scale
		shape[0, i, 2] = (shape[0, i, 2] - center[0, 2])* scale
	return shape
	
def decode_structure(model, feature, points_f, shape):
	"""
	segment shape
	"""
	global m
	n_points = shape.size(1)
	loc_f = model.pcEncoder(shape)
	if feature is None:
		feature = loc_f
	f_c1 = torch.cat([feature, loc_f], 1)
	label_prob = model.nodeClassifier(f_c1)
	_, label = torch.max(label_prob, 1)
	label = label.item()
	if label == 1 or label == 3:  # ADJ
		left, right = model.adjDecoder(f_c1)
		f_max, _ = torch.max(points_f, 2)
		f_c2 = torch.cat([f_max, f_c1], 1)
		point_label_prob, _ = model.decoder.loc_points_predictor(points_f, f_c2)
		point_label_prob = point_label_prob.cpu()
		_, point_label=torch.max(point_label_prob, 1)
		left_list=[]
		right_list=[]
		for i in range(n_points):
			if point_label[0, i].item() == 0:
				left_list.append(i)
			else:
				right_list.append(i)
		left_idx=torch.LongTensor(left_list).cuda()
		right_idx=torch.LongTensor(right_list).cuda()
		if left_idx.size(0) > 20 and right_idx.size(0) > 20:
			l_pc=torch.index_select(shape, 1, left_idx)
			l_feature=torch.index_select(points_f, 2, left_idx)
			r_pc=torch.index_select(shape, 1, right_idx)
			r_feature=torch.index_select(points_f, 2, right_idx)
			l_labeling = decode_structure(model, left, l_feature, l_pc)
			r_labeling = decode_structure(model, right, r_feature, r_pc)
			prediction = torch.LongTensor(n_points).zero_()
			for i, j in enumerate(left_idx):
				prediction[j.item()]=l_labeling[i]
			for i, j in enumerate(right_idx):
				prediction[j.item()]=r_labeling[i]
			return prediction
		else:
			prediction = torch.LongTensor(n_points).fill_(m)
			m += 1
			return prediction
	elif label == 2:
		f_max, _ = torch.max(points_f, 2)
		f_c2 = torch.cat([f_max, f_c1], 1)
		point_label_prob, _ = model.decoder.loc_points_predictor_multi(points_f, f_c2)
		point_label_prob = point_label_prob.cpu()
		_, point_label = torch.max(point_label_prob, 1)
		prediction = torch.LongTensor(n_points)
		for i in range(point_label.size(1)):
			prediction[i] = m+point_label[0, i].item()
		mx = torch.max(point_label).item()
		m += mx + 1
		return prediction
	else:
		prediction=torch.LongTensor(n_points).fill_(m)
		m += 1
		return prediction
	
m = 0
def main():
	config = util.get_args()
	config.cuda = not config.no_cuda
	torch.cuda.set_device(config.gpu)
	if config.cuda and torch.cuda.is_available():
		print("Using CUDA on GPU ", config.gpu)
	else:
		print("Not using CUDA.")
	net = partnet_model.PARTNET(config)
	net.load_state_dict(torch.load(config.save_path + '/partnet_final.pkl', map_location=lambda storage, loc: storage.cuda(config.gpu)))
	if config.cuda:
		net.cuda()
	net.eval()
	
	if not os.path.exists(config.output_path + 'segmented'):
		os.makedirs(config.output_path + 'segmented')
	print("Loading data ...... ", end='\n', flush=True)
	
	shape = torch.from_numpy(sio.loadmat(config.data_path + 'demo.mat')['pc']).float()
	##for your own new shape
	##shape = normalize_shape(shape)
	with torch.no_grad():
		shape = shape.cuda()
		points_feature = net.pointnet(shape)
		root_feature = net.pcEncoder(shape)
		global m
		m = 0 
		label = decode_structure(net, root_feature, points_feature, shape)
		
		#segmented results
		writeply(config.output_path + 'segmented/demo.ply', shape, label)
		print('Successfully output result!')
			
if __name__ == '__main__':
	main()
