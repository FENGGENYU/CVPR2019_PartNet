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


def evaluate(pred_grp, gt_grp, seg, n_category=4, at=0.5):
	total_sgpn = np.zeros(n_category)
	tpsins = [[] for itmp in range(n_category)]
	fpsins = [[] for itmp in range(n_category)]
	pts_in_pred = [[] for itmp in range(n_category)]
	pts_in_gt = [[] for itmp in range(n_category)]
	un = np.unique(pred_grp)
	for i, g in enumerate(un):
		tmp = (pred_grp == g)
		sem_seg_g = int(stats.mode(seg[tmp])[0])
		pts_in_pred[sem_seg_g] += [tmp]
	un = np.unique(gt_grp)
	for ig, g in enumerate(un):
		tmp = (gt_grp == g)
		sem_seg_g = int(stats.mode(seg[tmp])[0])
		pts_in_gt[sem_seg_g] += [tmp]
		total_sgpn[sem_seg_g] += 1
	for i_sem in range(n_category):
		tp = [0.] * len(pts_in_pred[i_sem])
		fp = [0.] * len(pts_in_pred[i_sem])
		gtflag = np.zeros(len(pts_in_gt[i_sem]))

		for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
			ovmax = -1.

			for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
				union = (ins_pred | ins_gt)
				intersect = (ins_pred & ins_gt)
				iou = float(np.sum(intersect)) / np.sum(union)

				if iou > ovmax:
					ovmax = iou
					igmax = ig

			if ovmax >= at:
				if gtflag[igmax] == 0:
					tp[ip] = 1 # true
					gtflag[igmax] = 1
				else:
					fp[ip] = 1 # multiple det
			else:
				fp[ip] = 1 # false positive
		tpsins[i_sem] += tp
		fpsins[i_sem] += fp
	return tpsins, fpsins, total_sgpn


def eval_3d_perclass(tp, fp, npos):
	tp = np.asarray(tp).astype(np.float)
	fp = np.asarray(fp).astype(np.float)
	tp = np.cumsum(tp)
	fp = np.cumsum(fp)
	rec = tp / npos
	prec = tp / (fp+tp)

	ap = 0.
	for t in np.arange(0, 1, 0.1):
		prec1 = prec[rec>=t]
		prec1 = prec1[~np.isnan(prec1)]
		if len(prec1) == 0:
			p = 0.
		else:
			p = max(prec1)
			if not p:
				p = 0.
		ap = ap + p / 10
	return ap, rec, prec


def decode_structure(model, feature, points_f, shape):
	"""
	Decode a root code into a tree structure of boxes
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
		bbox, _ = model.decoder.loc_points_predictor(points_f, f_c2)
		bbox = bbox.cpu()
		_, point_label=torch.max(bbox, 1)
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
		bbox, _ = model.decoder.loc_points_predictor_multi(points_f, f_c2)
		bbox = bbox.cpu()
		_, point_label = torch.max(bbox, 1)
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
	
	if not os.path.exists(config.output_path + 'gt_grp'):
		os.makedirs(config.output_path + 'gt_grp')
	if not os.path.exists(config.output_path + 'gt'):
		os.makedirs(config.output_path + 'gt')
	if not os.path.exists(config.output_path + 'segmented'):
		os.makedirs(config.output_path + 'segmented')
	print("Loading data ...... ", end='\n', flush=True)
	data_loader_batch = Data_Loader(config.data_path, config.training, config.split_num, config.total_num)
	NUM_CATEGORY = config.label_category
	recall_all = 0
	with torch.no_grad():
		tpsins = [[] for itmp in range(NUM_CATEGORY)]
		fpsins = [[] for itmp in range(NUM_CATEGORY)]
		total_sgpn = np.zeros(NUM_CATEGORY)
		tpsins_2 = [[] for itmp in range(NUM_CATEGORY)]
		fpsins_2 = [[] for itmp in range(NUM_CATEGORY)]
		total_sgpn_2 = np.zeros(NUM_CATEGORY)
		bad_shape = torch.zeros(len(data_loader_batch), 1)
		bad_num = 0
		for n in range(len(data_loader_batch)):
			shape = data_loader_batch[n].shape.cuda()
			points_feature = net.pointnet(shape)
			root_feature = net.pcEncoder(shape)
			print('index : ', n)
			global m
			m = 0 
			label = decode_structure(net, root_feature, points_feature, shape)
			seg_gt = data_loader_batch[n].shape_label
			grp_gt = data_loader_batch[n].grp
			
			#ground truth fine-grained groups
			writeply(config.output_path + 'gt_grp/%d.ply' % (n), data_loader_batch[n].shape, grp_gt)
			#ground truth semantic group
			writeply(config.output_path + 'gt/%d.ply' % (n), data_loader_batch[n].shape, seg_gt)
			#segmented results
			writeply(config.output_path + 'segmented/%d.ply' % (n), data_loader_batch[n].shape, label)
			tp, fp, groups = evaluate(label.numpy(), grp_gt.numpy(), seg_gt.numpy(), NUM_CATEGORY, at=0.25)
			tp2, fp2, groups2 = evaluate(label.numpy(), grp_gt.numpy(), seg_gt.numpy(), NUM_CATEGORY, at=0.5)
			
			for i in range(NUM_CATEGORY):
				tpsins[i] += tp[i]
				fpsins[i] += fp[i]
				total_sgpn[i] += groups[i]
				tpsins_2[i] += tp2[i]
				fpsins_2[i] += fp2[i]
				total_sgpn_2[i] += groups2[i]

		ap = np.zeros(NUM_CATEGORY)
		ap2 = np.zeros(NUM_CATEGORY)
		for i_sem in range(NUM_CATEGORY):
			ap[i_sem], _, _ = eval_3d_perclass(tpsins[i_sem], fpsins[i_sem], total_sgpn[i_sem])
			ap2[i_sem], _, _ = eval_3d_perclass(tpsins_2[i_sem], fpsins_2[i_sem], total_sgpn_2[i_sem])
			
		print('Instance Segmentation AP(IoU 0.25):', ap)
		print('Instance Segmentation mAP(IoU 0.25:', np.mean(ap))
		print('Instance Segmentation AP(IoU 0.5):', ap2)
		print('Instance Segmentation mAP(IoU 0.5):', np.mean(ap2))
			
if __name__ == '__main__':
	main()
