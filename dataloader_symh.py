import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
from torch.autograd import Variable
import math
from pytorch_ops.sampling.sample import FarthestSample
from pytorch_ops.losses.cd.cd import CDModule

m_grp = 0

def vrrotvec2mat(rotvector, angle):
	s = math.sin(angle)
	c = math.cos(angle)
	t = 1 - c
	x = rotvector[0]
	y = rotvector[1]
	z = rotvector[2]
	m = torch.FloatTensor(
		[[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
		 [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
		 [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])
	return m

#segmentation for symmetric node
def multilabel(points, shape, cdloss):
	c = torch.LongTensor(1, 2048).zero_()
	c = c - 1
	for i in range(points.size(0)):
		a = points[i].unsqueeze(0).cuda()
		_, index, _, _ = cdloss(a, shape)
		b = torch.unique(index.cpu())
		for k in range(b.size(0)):
			c[0, b[k].item()] = i
	return c
		
class Tree(object):
	class NodeType(Enum):
		LEAF = 0	 # leaf node
		ADJ = 1	 # adjacency (adjacent part assembly) node
		SYM = 2	 # symmetry (symmetric part grouping) node
		SYM_ADJ = 3 #reflect

	class Node(object):
		def __init__(self,
					 leaf_points=None,
					 left=None,
					 right=None,
					 node_type=None,
					 sym_p=None,
					 sym_a=None,
					 sym_t=None,
					 semantic_label=None):
			self.leaf_points = leaf_points	# node points
			if isinstance(sym_t, int):
				self.sym_t = torch.LongTensor([sym_t])
			else:
				self.sym_t = None
			if isinstance(sym_a, int):
				self.sym_a = torch.LongTensor([sym_a])
			else:
				self.sym_a = None
			self.sym_p = sym_p
			self.sym_type = self.sym_a
			self.left = left  # left child for ADJ or SYM (a symmeter generator)
			self.right = right	# right child
			self.node_type = node_type
			self.label = torch.LongTensor([self.node_type.value])
			self.is_root = False
			self.semantic_label = semantic_label

		def is_leaf(self):
			return self.node_type == Tree.NodeType.LEAF

		def is_adj(self):
			return self.node_type == Tree.NodeType.ADJ

		def is_sym(self):
			return self.node_type == Tree.NodeType.SYM
		
		def is_sym_adj(self):
			return self.node_type == Tree.NodeType.SYM_ADJ

	def __init__(self, parts, ops, syms, labels, shape):
		parts_list = [p for p in torch.split(parts, 1, 0)]
		sym_param = [s for s in torch.split(syms, 1, 0)]
		part_labels = [s for s in torch.split(labels, 1, 0)]
		parts_list.reverse()
		sym_param.reverse()
		part_labels.reverse()
		queue = []
		sym_node_num = 0
		for id in range(ops.size()[1]):
			if ops[0, id] == Tree.NodeType.LEAF.value:
				queue.append(
					Tree.Node(leaf_points=parts_list.pop(), node_type=Tree.NodeType.LEAF, semantic_label=part_labels.pop()))
			elif ops[0, id] == Tree.NodeType.ADJ.value:
				left_node = queue.pop()
				right_node = queue.pop()
				queue.append(
					Tree.Node(
						left=left_node,
						right=right_node,
						node_type=Tree.NodeType.ADJ))
			elif ops[0, id] == Tree.NodeType.SYM.value:
				node = queue.pop()
				s = sym_param.pop()
				b = s[0, 0] + 1
				t = s[0, 7].item()
				p = s[0, 1:7]
				if t > 0:
					t = round(1.0/t)
				queue.append(
					Tree.Node(
						left=node,
						sym_p=p.unsqueeze(0),
						sym_a=int(b),
						sym_t=int(t),
						node_type=Tree.NodeType.SYM))
				if b != 1:
					sym_node_num += 1
		assert len(queue) == 1
		self.root = queue[0]
		self.root.is_root = True
		assert self.root.is_adj()
		self.shape = shape
		if sym_node_num == 0:
			self.n_syms = torch.Tensor([sym_node_num]).cuda()
		else:
			self.n_syms = torch.Tensor([1/sym_node_num]).cuda()

#find GT label's index in input
def Attention(feature2048, shape):
	index = []
	for i in range(shape.size(1)):
		if feature2048[0, i] > -1:
			index.append(i)
	pad_index = []
	while len(pad_index) < 2048:
		pad_index.extend(index)
	pad_index = torch.LongTensor(pad_index[:2048])
	return pad_index.unsqueeze(0).cpu()
	
#construct groundtruth for pointcloud segmentation

def dfs_fix(node, shape, cdloss, shape_normal, seg, grp, reflect=None):
	global m_grp
	if node.is_leaf():
		# find node's corresponding points on input
		_, index, _ , _	 = cdloss(node.leaf_points[:, :, :3].cuda(), shape)
		b = torch.unique(index.cpu())
		c = torch.LongTensor(1, 2048).zero_()
		c = c - 1
		for i in range(b.size(0)):
			c[0, b[i].item()] = 0
		node.index = c #segmentation GT binary label
		idx = Attention(c, shape) #node's corresponding idx
		#node's corresponding points
		node.points = torch.index_select(shape_normal, 1, idx.squeeze(0).long().cpu())
		#node's corresponding idx
		node.pad_index = idx
		for i in range(node.pad_index.size(1)):
			seg[node.pad_index[0, i].item()] = node.semantic_label
			grp[node.pad_index[0, i].item()] = m_grp
		m_grp += 1
		if reflect is not None:
			#recover reflect's children
			re_leaf_points = torch.cat([node.leaf_points[:, :, :3], node.leaf_points[:, :, :3]+node.leaf_points[:, :, 3:]], 1)
			re_leaf_points = re_leaf_points.squeeze(0).cpu()
			sList = torch.split(reflect, 1, 0)
			ref_normal = torch.cat([sList[0], sList[1], sList[2]])
			ref_normal = ref_normal / torch.norm(ref_normal)
			ref_point = torch.cat([sList[3], sList[4], sList[5]])
			new_points = 2 * ref_point.add(-re_leaf_points).matmul(ref_normal)
			new_points = new_points.unsqueeze(-1)
			new_points = new_points.repeat(1, 3)
			new_points = ref_normal.mul(new_points).add(re_leaf_points)
			new_points = torch.cat([new_points[:2048, :], new_points[2048:, :] - new_points[:2048, :]], 1)
			New_node = Tree.Node(leaf_points=new_points.unsqueeze(0), node_type=Tree.NodeType.LEAF)
			#build node for reflect node's children
			_, index, _ , _	 = cdloss(New_node.leaf_points[:, :, :3].cuda(), shape)
			b = torch.unique(index.cpu())
			reflect_c = torch.LongTensor(1, 2048).zero_()
			reflect_c = reflect_c - 1
			for i in range(b.size(0)):
				reflect_c[0, b[i].item()] = 0
			New_node.index = reflect_c
			idx = Attention(reflect_c, shape)
			New_node.points = torch.index_select(shape_normal, 1, idx.squeeze(0).long().cpu())
			New_node.pad_index = idx
			New_node.semantic_label = node.semantic_label
			for i in range(New_node.pad_index.size(1)):
				seg[New_node.pad_index[0, i].item()] = New_node.semantic_label
				grp[New_node.pad_index[0, i].item()] = m_grp
			m_grp += 1
			return torch.Tensor([0]).cuda(), New_node
		else:
			return torch.Tensor([0]).cuda(), node
	if node.is_adj():
		l_num, new_node_l = dfs_fix(node.left, shape, cdloss, shape_normal, seg, grp, reflect)
		r_num, new_node_r = dfs_fix(node.right, shape, cdloss, shape_normal, seg, grp, reflect)
		#build adj node
		c = torch.LongTensor(1, 2048).zero_()
		c = c - 1
		for i in range(2048):
			if node.left.index[0, i].item() > -1:
				c[0, i] = 0
		for i in range(2048):
			if node.right.index[0, i].item() > -1:
				c[0, i] = 1
		node.index = c
		idx = Attention(c, shape)
		node.points = torch.index_select(shape_normal, 1, idx.squeeze(0).long().cpu())
		node.pad_index = idx
		
		if reflect is not None:
			New_node = Tree.Node(left=new_node_l, right=new_node_r, node_type=Tree.NodeType.ADJ)
			reflect_c = torch.LongTensor(1, 2048).zero_()
			reflect_c = reflect_c - 1
			for i in range(2048):
				if new_node_l.index[0, i].item() > -1:
					reflect_c[0, i] = 0
			for i in range(2048):
				if new_node_r.index[0, i].item() > -1:
					reflect_c[0, i] = 1
			New_node.index = reflect_c
			idx = Attention(reflect_c, shape)
			New_node.points = torch.index_select(shape_normal, 1, idx.squeeze(0).long().cpu())
			New_node.pad_index = idx
			
			return l_num + r_num + torch.Tensor([2]).cuda(), New_node
		else:
			return l_num + r_num + torch.Tensor([1]).cuda(), node
	if node.is_sym():
		#build symmetric node
		t = node.sym_t.item()
		p = node.sym_p.squeeze(0)
		
		if node.sym_type.item() == 1: #reflect node
			child_num, new_node = dfs_fix(node.left, shape, cdloss, shape_normal, seg, grp, p)
			
			c = torch.LongTensor(1, 2048).zero_()
			c = c - 1
			for i in range(2048):
				if node.left.index[0, i].item() > -1:
					c[0, i] = 0
			for i in range(2048):
				if new_node.index[0, i].item() > -1:
					c[0, i] = 1
			node.index = c
			node.right = new_node
			idx = Attention(c, shape)
			node.points = torch.index_select(shape_normal, 1, idx.squeeze(0).long().cpu())
			node.node_type = Tree.NodeType.SYM_ADJ
			node.label = torch.LongTensor([node.node_type.value])
			node.pad_index = idx
			
			return child_num + torch.Tensor([1]).cuda(), node
		else:
			child_num, _= dfs_fix(node.left, shape, cdloss, shape_normal, seg, grp, None)
			new_leaf_points = node.left.leaf_points.squeeze(0)
			leaf_points_list = [new_leaf_points.unsqueeze(0)]
			
			new_leaf_points = torch.cat([new_leaf_points[:, :3] , new_leaf_points[:, :3] + new_leaf_points[:, 3:]], 0)
			
			if node.sym_type.item() == 0:#rotate symmetry
				sList = torch.split(p, 1, 0)
				f1 = torch.cat([sList[0], sList[1], sList[2]])
				if f1[1] < 0:
					f1 = - f1
				f1 = f1 / torch.norm(f1)
				f2 = torch.cat([sList[3], sList[4], sList[5]])
				folds = int(t)
				a = 1.0 / float(folds)
				for i in range(folds - 1):
					angle = a * 2 * 3.1415 * (i + 1)
					rotm = vrrotvec2mat(f1, angle)
					sym_leaf_points = rotm.matmul(new_leaf_points.add(-f2).t()).t().add(f2)
					sym_leaf_points = torch.cat([sym_leaf_points[:2048, :] , sym_leaf_points[2048:, :] - sym_leaf_points[:2048, :]], 1)
					leaf_points_list.append(sym_leaf_points.unsqueeze(0))
			elif node.sym_type.item() == 2: #translate symmetry
				sList = torch.split(p, 1, 0)
				trans = torch.cat([sList[0], sList[1], sList[2]])
				folds = t - 1
				for i in range(folds):
					sym_leaf_points = new_leaf_points.add(trans.mul(i + 1))
					sym_leaf_points = torch.cat([sym_leaf_points[:2048, :] , sym_leaf_points[2048:, :] - sym_leaf_points[:2048, :]], 1)
					leaf_points_list.append(sym_leaf_points.unsqueeze(0))

			a = torch.cat(leaf_points_list, 0)
			node.index = multilabel(a[:, :, :3], shape, cdloss)
			idx = Attention(node.index, shape)
			node.points = torch.index_select(shape_normal, 1, idx.squeeze(0).long().cpu())
			node.pad_index = Attention(node.index, shape)
			for i in range(node.pad_index.size(1)):
				seg[node.pad_index[0, i].item()] = node.left.semantic_label
			for i in range(2048):
				if node.index[0, i].item() > -1:
					grp[i] = m_grp + node.index[0, i]
			m_grp = m_grp + torch.max(node.index) + 1
			return torch.Tensor([1]).cuda(), node
		
class Data_Loader(data.Dataset):
	def __init__(self, dir, is_train, split_num, total_num):
		self.dir = dir
		op_data = torch.from_numpy(loadmat(self.dir + 'training_trees/ops.mat')['ops']).int()
		label_data = torch.from_numpy(loadmat(self.dir + 'training_trees/labels.mat')['labels']).int()
		sym_data = torch.from_numpy(loadmat(self.dir + 'training_trees/syms.mat')['syms']).float()
		num_examples = op_data.size()[1]
		op_data = torch.chunk(op_data, num_examples, 1)
		label_data = torch.chunk(label_data, num_examples, 1)
		sym_data = torch.chunk(sym_data, num_examples, 1)
		self.trees = []
		self.training = is_train
		if is_train:
			begin = 0
			end = split_num 
		else:
			begin = split_num
			end = total_num
		for i in range(begin, end):
			parts = torch.from_numpy(loadmat(self.dir + 'training_data_models_segment_2048_normals/%d.mat' % i)['pc']).float()
			shape = torch.from_numpy(loadmat(self.dir + 'models_2048_points_normals/%d.mat' % i)['pc']).float()
			ops = torch.t(op_data[i])
			syms = torch.t(sym_data[i])
			labels = torch.t(label_data[i])
			tree = Tree(parts, ops, syms, labels, shape)
			cdloss = CDModule()
			seg = torch.LongTensor(2048).zero_() # for ap calculation
			grp = torch.LongTensor(2048).zero_()
			global m_grp
			m_grp = 0
			num_node, _ = dfs_fix(tree.root, shape[0, :, :3].unsqueeze(0).cuda(), cdloss, shape, seg, grp)
			tree.n_nodes = num_node
			tree.shape_label = seg
			tree.grp = grp
			self.trees.append(tree)
			print('load data', i)
		print(len(self.trees))

	def __getitem__(self, index):
		tree = self.trees[index]
		return tree

	def __len__(self):
		return len(self.trees)
