import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_ops.grouping.group import QueryBallPoint, GroupPoints
# from pytorch_ops.sampling.sample import SampleFunction
# from pytorch_ops.interpolation.interpolate import InterpolateFunction


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
		l0_xyz = input_data[:, :, :3].permute(0, 2, 1)
		l0_points = input_data[:, :, 3:].permute(0, 2, 1)
		l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)	 # 128
		l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)	 # 256
		l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)	 # 128

		l2_points = self.fp1(l2_xyz, l3_xyz, l2_points, l3_points)
		l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
		l0_points = self.fp3(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
		#l0_points = l0_points.permute(0, 2, 1)
		return l0_points

def timeit(tag, t):
	print("{}: {}s".format(tag, time() - t))
	return time()

def pc_normalize(pc):
	l = pc.shape[0]
	centroid = np.mean(pc, axis=0)
	pc = pc - centroid
	m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
	pc = pc / m
	return pc

def square_distance(src, dst):
	"""
	Calculate Euclid distance between each two points.
	src^T * dst = xn * xm + yn * ym + zn * zm；
	sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
	sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
	dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
		 = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
	Input:
		src: source points, [B, N, C]
		dst: target points, [B, M, C]
	Output:
		dist: per-point square distance, [B, N, M]
	"""
	B, N, _ = src.shape
	_, M, _ = dst.shape
	dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
	dist += torch.sum(src ** 2, -1).view(B, N, 1)
	dist += torch.sum(dst ** 2, -1).view(B, 1, M)
	return dist


def index_points(points, idx):
	"""
	Input:
		points: input points data, [B, N, C]
		idx: sample index data, [B, S]
	Return:
		new_points:, indexed points data, [B, S, C]
	"""
	device = points.device
	B = points.shape[0]
	view_shape = list(idx.shape)
	view_shape[1:] = [1] * (len(view_shape) - 1)
	repeat_shape = list(idx.shape)
	repeat_shape[0] = 1
	batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
	new_points = points[batch_indices, idx, :]
	return new_points


def farthest_point_sample(xyz, npoint):
	"""
	Input:
		xyz: pointcloud data, [B, N, 3]
		npoint: number of samples
	Return:
		centroids: sampled pointcloud index, [B, npoint]
	"""
	device = xyz.device
	B, N, C = xyz.shape
	centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
	distance = torch.ones(B, N).to(device) * 1e10
	farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
	batch_indices = torch.arange(B, dtype=torch.long).to(device)
	for i in range(npoint):
		centroids[:, i] = farthest
		centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
		dist = torch.sum((xyz - centroid) ** 2, -1)
		mask = dist < distance
		distance[mask] = dist[mask]
		farthest = torch.max(distance, -1)[1]
	return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
	"""
	Input:
		radius: local region radius
		nsample: max sample number in local region
		xyz: all points, [B, N, 3]
		new_xyz: query points, [B, S, 3]
	Return:
		group_idx: grouped points index, [B, S, nsample]
	"""
	device = xyz.device
	B, N, C = xyz.shape
	_, S, _ = new_xyz.shape
	group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
	sqrdists = square_distance(new_xyz, xyz)
	group_idx[sqrdists > radius ** 2] = N
	group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
	group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
	mask = group_idx == N
	group_idx[mask] = group_first[mask]
	return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
	"""
	Input:
		npoint:
		radius:
		nsample:
		xyz: input points position data, [B, N, 3]
		points: input points data, [B, N, D]
	Return:
		new_xyz: sampled points position data, [B, npoint, nsample, 3]
		new_points: sampled points data, [B, npoint, nsample, 3+D]
	"""
	B, N, C = xyz.shape
	S = npoint
	fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
	new_xyz = index_points(xyz, fps_idx)
	idx = query_ball_point(radius, nsample, xyz, new_xyz)
	grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
	grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

	if points is not None:
		grouped_points = index_points(points, idx)
		new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
	else:
		new_points = grouped_xyz_norm
	if returnfps:
		return new_xyz, new_points, grouped_xyz, fps_idx
	else:
		return new_xyz, new_points


def sample_and_group_all(xyz, points):
	"""
	Input:
		xyz: input points position data, [B, N, 3]
		points: input points data, [B, N, D]
	Return:
		new_xyz: sampled points position data, [B, 1, 3]
		new_points: sampled points data, [B, 1, N, 3+D]
	"""
	device = xyz.device
	B, N, C = xyz.shape
	new_xyz = torch.zeros(B, 1, C).to(device)
	grouped_xyz = xyz.view(B, 1, N, C)
	if points is not None:
		new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
	else:
		new_points = grouped_xyz
	return new_xyz, new_points


class pointnet_sa_module(nn.Module):
	def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
		super(pointnet_sa_module, self).__init__()
		self.npoint = npoint
		self.radius = radius
		self.nsample = nsample
		# self.mlp_convs = nn.ModuleList()
		# self.mlp_bns = nn.ModuleList()
		last_channel = in_channel

		models = []
		for x in mlp:
			models.append(nn.Conv2d(last_channel, x, 1, 1))
			models.append(nn.BatchNorm2d(x))
			models.append(nn.ReLU())
			last_channel = x
		self.Model = nn.Sequential(*models)

		# for out_channel in mlp:
		#     self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
		#     self.mlp_bns.append(nn.BatchNorm2d(out_channel))
		#     last_channel = out_channel
		self.group_all = group_all

	def forward(self, xyz, points):
		"""
		Input:
			xyz: input points position data, [B, C, N]
			points: input points data, [B, D, N]
		Return:
			new_xyz: sampled points position data, [B, C, S]
			new_points_concat: sample points feature data, [B, D', S]
		"""
		xyz = xyz.permute(0, 2, 1)
		if points is not None:
			points = points.permute(0, 2, 1)

		if self.group_all:
			new_xyz, new_points = sample_and_group_all(xyz, points)
		else:
			new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
		# new_xyz: sampled points position data, [B, npoint, C]
		# new_points: sampled points data, [B, npoint, nsample, C+D]
		new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
		new_points = self.Model(new_points)
		new_points = torch.max(new_points, 2)[0]
		new_xyz = new_xyz.permute(0, 2, 1)
		return new_xyz, new_points


class pointnet_fp_module(nn.Module):
	def __init__(self, in_channel, mlp):
		super(pointnet_fp_module, self).__init__()
		# self.mlp_convs = nn.ModuleList()
		# self.mlp_bns = nn.ModuleList()
		channels = in_channel
		models = []
		for x in mlp:
			models.append(nn.Conv2d(channels, x, 1, 1))
			models.append(nn.BatchNorm2d(x))
			models.append(nn.ReLU())
			channels = x
		self.Model = nn.Sequential(*models)

	def forward(self, xyz1, xyz2, points1, points2):
		"""
		Input:
			xyz1: input points position data, [B, C, N]
			xyz2: sampled input points position data, [B, C, S]
			points1: input points data, [B, D, N]
			points2: input points data, [B, D, S]
		Return:
			new_points: upsampled points data, [B, D', N]
		"""
		xyz1 = xyz1.permute(0, 2, 1)
		xyz2 = xyz2.permute(0, 2, 1)

		points2 = points2.permute(0, 2, 1)
		B, N, C = xyz1.shape
		_, S, _ = xyz2.shape

		if S == 1:
			interpolated_points = points2.repeat(1, N, 1)
		else:
			dists = square_distance(xyz1, xyz2)
			dists, idx = dists.sort(dim=-1)
			dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

			dist_recip = 1.0 / (dists + 1e-8)
			norm = torch.sum(dist_recip, dim=2, keepdim=True)
			weight = dist_recip / norm
			interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

		if points1 is not None:
			points1 = points1.permute(0, 2, 1)
			new_points = torch.cat([points1, interpolated_points], dim=-1)
		else:
			new_points = interpolated_points

		new_points = new_points.permute(0, 2, 1)
		new_points = new_points.unsqueeze(-1)
		#print('new_points, ', new_points.shape)
		new_points = self.Model(new_points)

		new_points = new_points.squeeze(-1)
		return new_points