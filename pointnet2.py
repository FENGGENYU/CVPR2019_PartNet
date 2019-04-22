import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_ops.grouping.group import QueryBallPoint, GroupPoints
from pytorch_ops.sampling.sample import SampleFunction
from pytorch_ops.interpolation.interpolate import InterpolateFunction


class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.sa1 = pointnet_sa_module(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
		self.sa2 = pointnet_sa_module(npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 256], group_all=False)
		self.sa3 = pointnet_sa_module(npoint=None, radius=None, nsample=None, in_channel=259, mlp=[256, 128], group_all=True)

		self.fp1 = pointnet_fp_module(in_channel=256+128, mlp=[256, 256])
		self.fp2 = pointnet_fp_module(in_channel=128+256, mlp=[256, 128])
		self.fp3 = pointnet_fp_module(in_channel=6+128, mlp=[128, 128, 128])
		
	def forward(self, input_data):	# input:(b, n, 3)
		l0_xyz = input_data[:, :, :3]
		l0_points = input_data[:, :, 3:]
		l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)	 # 128
		l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)	 # 256
		l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)	 # 128

		l2_points = self.fp1(l2_xyz, l3_xyz, l2_points.transpose(1, 2), l3_points.transpose(1, 2))
		l1_points = self.fp2(l1_xyz, l2_xyz, l1_points.transpose(1, 2), l2_points)
		l0_points = self.fp3(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 2).transpose(1, 2), l1_points)

		return l0_points


class pointnet_sa_module(nn.Module):
	def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False, bn=True):
		super(pointnet_sa_module, self).__init__()
		self.sample_points = SampleFunction(npoint)
		self.radius = radius
		self.nsample = nsample
		self.group_all = group_all
		channels = in_channel
		models = []
		for x in mlp:
			models.append(nn.Conv2d(channels, x, 1, 1))
			models.append(nn.BatchNorm2d(x))
			models.append(nn.ReLU())
			channels = x
		self.Model = nn.Sequential(*models)
		print(self.Model)

	def forward(self, xyz, points):
		if self.group_all:
			new_xyz = torch.zeros(xyz.size()[0], 1, 3)
			if xyz.is_cuda:
				new_xyz = new_xyz.cuda()
			new_points = torch.cat([xyz, points], 2)
			new_points = new_points.unsqueeze(1)
		else:
			new_xyz, _ = self.sample_points(xyz)
			idx, pts_cnt = QueryBallPoint(self.radius, self.nsample)(xyz, new_xyz)
			grouped_xyz = GroupPoints()(xyz, idx)
			grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, self.nsample, 1)
			if points is not None:
				grouped_points = GroupPoints()(points, idx)
				new_points = torch.cat([grouped_xyz, grouped_points], -1)
			else:
				new_points = grouped_xyz
		new_points = new_points.permute(0, 3, 1, 2)

		new_points = self.Model(new_points)
		new_points = new_points.permute(0, 2, 3, 1)
		new_points, _ = torch.max(new_points, 2)
		return new_xyz, new_points


class pointnet_fp_module(nn.Module):
	def __init__(self, in_channel, mlp):
		super().__init__()
		self.convs = []
		channels = in_channel
		models = []
		for x in mlp:
			models.append(nn.Conv2d(channels, x, 1, 1))
			models.append(nn.BatchNorm2d(x))
			models.append(nn.ReLU())
			channels = x
		self.Model = nn.Sequential(*models)

	def forward(self, xyz1, xyz2, points1, points2):
		# xyz1:(b,n,3)
		# xyz2:(b,m,3)	 m < n
		# points1:(b,c1,n)
		# points2:(b,c2,m)
		# out:(b,mlp[-1],n)
		if xyz2.size(1) == 1:
			interpolate = points2.repeat(1, 1, xyz1.size(1))
		else:
			D = xyz1.unsqueeze(2) - xyz2.unsqueeze(1)
			D = torch.sum(torch.pow(D, 2), -1)
			dist, idx = torch.topk(D, 3, 2, False)
			idx = idx.int()
			dist = torch.clamp(dist, min=1e-10)
			norm = torch.sum(1.0/dist, 2, True).repeat(1, 1, 3)
			weight = (1.0/dist) / norm
			interpolate = InterpolateFunction()(points2, idx, weight)
		if points1 is not None:
			new_points1 = torch.cat([interpolate, points1], 1)	# B,nchannel1+nchannel2,ndataset1
		else:
			new_points1 = interpolate
		new_points1 = new_points1.unsqueeze(-1)
		new_points1 = self.Model(new_points1)
		return new_points1.squeeze(-1)
