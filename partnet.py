import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time
import torch.nn.functional as F
from pytorch_ops.losses.cd.cd import CDModule
from pytorch_ops.losses.emd.emd import EMDModule
from pytorch_ops.sampling.sample import FarthestSample
import pointnet2 as Pointnet
#########################################################################################
# Encoder
#########################################################################################


class PCEncoder(nn.Module):
	def __init__(self, bottleneck=128):
		super(PCEncoder, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, (1, 6), 1)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 128, 1, 1)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(128, 128, 1, 1)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(128, 256, 1, 1)
		self.bn4 = nn.BatchNorm2d(256)
		self.conv5 = nn.Conv2d(256, bottleneck, 1, 1)
		self.bn5 = nn.BatchNorm2d(bottleneck)
		self.dropout2d = nn.Dropout2d(p=0.2)

	def forward(self, input):
		input = input.unsqueeze(1)
		input = self.conv1(input)
		input = self.bn1(input)
		input = F.relu(input)
		input = self.conv2(input)
		input = self.bn2(input)
		input = F.relu(input)
		input = self.conv3(input)
		input = self.bn3(input)
		input = F.relu(input)
		input = self.conv4(input)
		input = self.dropout2d(input)
		input = self.bn4(input)
		input = F.relu(input)
		input = self.conv5(input)
		input = self.dropout2d(input)
		input = self.bn5(input)
		input = F.relu(input)
		input, _ = torch.max(input, 2)
		input = input.squeeze(-1)
		return input


#########################################################################################
# Decoder
#########################################################################################

class NodeClassifier(nn.Module):
	def __init__(self, feature_size, hidden_size):
		super(NodeClassifier, self).__init__()
		self.mlp1 = nn.Linear(feature_size*2, hidden_size)
		self.tanh = nn.Tanh()
		self.mlp2 = nn.Linear(hidden_size, 4)

	def forward(self, input_feature):
		output = self.mlp1(input_feature)
		output = self.tanh(output)
		output = self.mlp2(output)
		return output


class AdjDecoder(nn.Module):
	""" Decode an input (parent) feature into a left-child and a right-child feature """

	def __init__(self, feature_size, hidden_size):
		super(AdjDecoder, self).__init__()
		self.mlp = nn.Linear(feature_size*2, hidden_size)
		self.mlp_left = nn.Linear(hidden_size, feature_size)
		self.mlp_right = nn.Linear(hidden_size, feature_size)
		self.tanh = nn.Tanh()

	def forward(self, parent_feature):
		vector = self.mlp(parent_feature)
		vector = self.tanh(vector)
		left_feature = self.mlp_left(vector)
		left_feature = self.tanh(left_feature)
		right_feature = self.mlp_right(vector)
		right_feature = self.tanh(right_feature)
		return left_feature, right_feature


class SymDecoder(nn.Module):
	def __init__(self, feature_size, hidden_size):
		super(SymDecoder, self).__init__()
		self.mlp = nn.Linear(feature_size*2, hidden_size)  # layer for decoding a feature vector
		self.tanh = nn.Tanh()
		self.mlp_sg = nn.Linear(hidden_size, feature_size)	# layer for outputing the feature of symmetry generator
		self.mlp_sp = nn.Linear(hidden_size, hidden_size)  # layer for outputing the vector of symmetry parameter
		self.mlp_s1 = nn.Linear(hidden_size, 3)# symmetric label
		self.mlp_s2 = nn.Linear(hidden_size, 6)# symmetric parameter
		self.mlp_s3 = nn.Linear(hidden_size, 9)# max symmetric number

	def forward(self, parent_feature):
		vector = self.mlp(parent_feature)
		vector = self.tanh(vector)
		sym_gen_vector = self.mlp_sg(vector)
		sym_gen_vector = self.tanh(sym_gen_vector)
		sym_param_vector = self.mlp_sp(vector)
		sym_param_vector = self.tanh(sym_param_vector)
		sym_label_vector = self.mlp_s1(sym_param_vector)
		sym_vector = self.mlp_s2(sym_param_vector)
		sym_time_vector = self.mlp_s3(sym_param_vector)
		return sym_gen_vector, sym_label_vector, sym_vector, sym_time_vector

class PointPrediction(nn.Module):
	def __init__(self, feature_size=128):
		super().__init__()
		self.conv1 = nn.Conv2d(128+feature_size*3, 256, 1, 1)
		#self.bn1 = nn.BatchNorm2d(256)
		self.conv2 = nn.Conv2d(256, 128, 1, 1)
		#self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(128, 64, 1, 1)
		#self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 64, 1, 1)
		#self.bn4 = nn.BatchNorm2d(64)
		self.conv5 = nn.Conv2d(64, 2, 1, 1)
		#self.bn5 = nn.BatchNorm2d(2)
		self.relu = nn.ReLU()
		self.logsoftmax = nn.LogSoftmax(1)

	def forward(self, points_feature, inp_feature):
		#points_feature = points.transpose(1, 2)
		output = torch.cat([points_feature, inp_feature.unsqueeze(-1).repeat(1, 1, points_feature.size(2))], 1).unsqueeze(-1)
		output = self.conv1(output)
		#output = self.bn1(output)
		output = self.relu(output)
		output = self.conv2(output)
	   # output = self.bn2(output)
		output = self.relu(output)
		output = self.conv3(output)
		#output = self.bn3(output)
		output = self.relu(output)
		output = self.conv4(output)
		#output = self.bn4(output)
		output4 = self.relu(output)
		output = self.conv5(output4)
		#output = self.bn5(output)
		output = self.logsoftmax(output)
		maxpool, _ = torch.max(output4, 2)
		return output.squeeze(-1), maxpool.squeeze(-1)

class PointPredictionMulti(nn.Module):
	def __init__(self, feature_size=128):
		super().__init__()
		self.conv1 = nn.Conv2d(128+feature_size*3, 256, 1, 1)
		#self.bn1 = nn.BatchNorm2d(256)
		self.conv2 = nn.Conv2d(256, 128, 1, 1)
		#self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(128, 64, 1, 1)
		#self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 64, 1, 1)
		#self.bn4 = nn.BatchNorm2d(64)
		self.conv5 = nn.Conv2d(64, 10, 1, 1)
		#self.bn5 = nn.BatchNorm2d(2)
		self.relu = nn.ReLU()
		self.logsoftmax = nn.LogSoftmax(1)

	def forward(self, points_feature, inp_feature):
		#points_feature = points.transpose(1, 2)
		output = torch.cat([points_feature, inp_feature.unsqueeze(-1).repeat(1, 1, points_feature.size(2))], 1).unsqueeze(-1)
		output = self.conv1(output)
		#output = self.bn1(output)
		output = self.relu(output)
		output = self.conv2(output)
	   # output = self.bn2(output)
		output = self.relu(output)
		output = self.conv3(output)
		#output = self.bn3(output)
		output = self.relu(output)
		output = self.conv4(output)
		#output = self.bn4(output)
		output4 = self.relu(output)
		output = self.conv5(output4)
		#output = self.bn5(output)
		output = self.logsoftmax(output)
		maxpool, _ = torch.max(output4, 2)
		return output.squeeze(-1), maxpool.squeeze(-1)

class PARTNETDecoder(nn.Module):
	def __init__(self, config, decoder_param_path=None):
		super(PARTNETDecoder, self).__init__()
		self.loc_points_predictor = PointPrediction(feature_size=config.feature_size)
		self.loc_points_predictor_multi = PointPredictionMulti(feature_size=config.feature_size)
		self.pc_encoder = PCEncoder(bottleneck=128)
		self.adj_decoder = AdjDecoder(feature_size=config.feature_size, hidden_size=config.hidden_size)
		self.sym_adj_decoder = AdjDecoder(feature_size=config.feature_size, hidden_size=config.hidden_size)
		self.sym_decoder = SymDecoder(feature_size=config.feature_size, hidden_size=config.hidden_size)
		self.node_classifier = NodeClassifier(feature_size=config.feature_size, hidden_size=config.hidden_size)

class PARTNET(nn.Module):
	def __init__(self, config, encoder_param_path=None, decoder_param_path=None):
		super(PARTNET, self).__init__()
		self.pointnet = Pointnet.Encoder()
		self.decoder = PARTNETDecoder(config, decoder_param_path)
		# pytorch's mean squared error loss
		self.mseLoss = nn.MSELoss(reduce=False)
		self.nllloss = nn.NLLLoss(reduce=False)
		self.creLoss = nn.CrossEntropyLoss(reduce=False)
		self.cdloss = CDModule()
		self.emdloss = EMDModule()
		self.sample = FarthestSample(256)

	def pcEncoder(self, points):
		return self.decoder.pc_encoder(points)

	def adjDecoder(self, feature):
		return self.decoder.adj_decoder(feature)

	def symadjDecoder(self, feature):
		return self.decoder.sym_adj_decoder(feature)
		
	def symDecoder(self, feature):
		return self.decoder.sym_decoder(feature)

	def nodeClassifier(self, feature):
		return self.decoder.node_classifier(feature)

	def symTimeLossEstimator(self, sym_time, gt_sym_time):
		return self.creLoss(sym_time, gt_sym_time).mul_(30)

	def symLabelLossEstimator(self, sym_label, gt_sym_label):
		return self.creLoss(sym_label, gt_sym_label).mul_(30)

	def symLossEstimator(self, sym_param, gt_sym_param):
		return torch.mean(self.mseLoss(sym_param, gt_sym_param).mul_(30), 1)

	def classifyLossEstimator(self, label_vector, gt_label_vector):
		a = self.creLoss(label_vector, gt_label_vector).mul(30)	 # 20
		return a
		
	def vectorAdder(self, v1, v2):
		return v1.add_(v2)

	def vectorAdder3(self, v1, v2, v3):
		return v1.add_(v2).add_(v3)

	def vectorAdder4(self, v1, v2, v3, v4):
		return v1.add_(v2).add_(v3).add_(v4)

	def vectorzero(self):
		temp = torch.zeros(1)
		return Variable(temp.cuda())

	def feature_concat2(self, f1, f2):
		return torch.cat([f1, f2], 1)

	def feature_concat3(self, f1, f2, f3):
		return torch.cat([f1, f2, f3], 1)

	def locPointsPredic(self, shape, feature, pad_index, gt):
		newf = []
		newl = []
		for i in range(pad_index.size(0)):
			new_feature2= torch.index_select(shape[i].unsqueeze(0), 2, pad_index[i])
			new_node_label2 = torch.index_select(gt[i].unsqueeze(0), 1, pad_index[i])
			newf.append(new_feature2)
			newl.append(new_node_label2)
		
		newf = torch.cat(newf, 0)
		newf_max, _ = torch.max(newf, 2)
		feature = torch.cat([newf_max, feature], 1)
		gene_label, last2feature = self.decoder.loc_points_predictor(newf, feature)
		newl = torch.cat(newl, 0)
		
		loss = torch.mean(self.nllloss(gene_label, newl), 1).mul_(30)
		_, index = torch.max(gene_label, 1)
		acc = torch.sum(torch.eq(index, newl), 1).float()
		return loss, acc, last2feature

	def locPointsPredic_multi(self, shape, feature, pad_index, gt):
		newf = []
		newl = []
		for i in range(pad_index.size(0)):
			new_feature2= torch.index_select(shape[i].unsqueeze(0), 2, pad_index[i])
			new_node_label2 = torch.index_select(gt[i].unsqueeze(0), 1, pad_index[i])
			newf.append(new_feature2)
			newl.append(new_node_label2)
		
		newf = torch.cat(newf, 0)
		newf_max, _ = torch.max(newf, 2)
		feature = torch.cat([newf_max, feature], 1)
		gene_label, last2feature = self.decoder.loc_points_predictor_multi(newf, feature)
		newl = torch.cat(newl, 0)
		
		loss = torch.mean(self.nllloss(gene_label, newl), 1).mul_(30)
		_, index = torch.max(gene_label, 1)
		acc = torch.sum(torch.eq(index, newl), 1).float()
		
		return loss, acc, last2feature

def jitter(shape):
	input_data = shape
	jitter_points = torch.randn(input_data.size())
	jitter_points = torch.clamp(0.01*jitter_points, min=-0.05, max=0.05)
	jitter_points += input_data
	return jitter_points
	
def decode_structure_fold(fold, tree, points_f):
	def decode_node(node, feature):
		if node.is_leaf():
			input_data = jitter(node.points)
			local_f = fold.add('pcEncoder', input_data)
			feature_c = fold.add('feature_concat2', feature, local_f)
			label = fold.add('nodeClassifier', feature_c)
			label_loss = fold.add('classifyLossEstimator', label, node.label)
			return label_loss, fold.add('vectorzero'), fold.add('vectorzero')

		elif node.is_adj():
			input_data = jitter(node.points)
			local_f = fold.add('pcEncoder', input_data)
			feature_c = fold.add('feature_concat2', feature, local_f)
			node_segloss, acc, last2feature = fold.add('locPointsPredic', points_f, feature_c, node.pad_index, node.index).split(3)
			left, right = fold.add('adjDecoder', feature_c).split(2)
			left_label_loss, left_segloss, left_acc = decode_node(node.left, left)
			right_label_loss, right_segloss, right_acc = decode_node(node.right, right)
			label = fold.add('nodeClassifier', feature_c)
			label_loss = fold.add('classifyLossEstimator', label, node.label)
			child_label_loss = fold.add('vectorAdder', left_label_loss, right_label_loss)
			node_label_loss = fold.add('vectorAdder', child_label_loss, label_loss)
			child_segloss = fold.add('vectorAdder', left_segloss, right_segloss)
			child_acc = fold.add('vectorAdder', left_acc, right_acc)
			node_segloss = fold.add('vectorAdder', child_segloss, node_segloss)
			acc = fold.add('vectorAdder', acc, child_acc) 
			return node_label_loss, node_segloss, acc

		elif node.is_sym():
			input_data = jitter(node.points)
			local_f = fold.add('pcEncoder', input_data)
			feature_c = fold.add('feature_concat2', feature, local_f)
			node_segloss, acc, last2feature = fold.add('locPointsPredic_multi', points_f, feature_c, node.pad_index, node.index).split(3)
			label = fold.add('nodeClassifier', feature_c)
			node_label_loss = fold.add('classifyLossEstimator', label, node.label)
			return node_label_loss, node_segloss, acc
		
		elif node.is_sym_adj():
			input_data = jitter(node.points)
			local_f = fold.add('pcEncoder', input_data)
			feature_c = fold.add('feature_concat2', feature, local_f)
			node_segloss, acc, last2feature = fold.add('locPointsPredic', points_f, feature_c, node.pad_index, node.index).split(3)
			left, right = fold.add('adjDecoder', feature_c).split(2)
			left_label_loss, left_segloss, left_acc = decode_node(node.left, left)
			right_label_loss, right_segloss, right_acc = decode_node(node.right, right)
			label = fold.add('nodeClassifier', feature_c)
			label_loss = fold.add('classifyLossEstimator', label, node.label)
			child_label_loss = fold.add('vectorAdder', left_label_loss, right_label_loss)
			node_label_loss = fold.add('vectorAdder', child_label_loss, label_loss)
			child_segloss = fold.add('vectorAdder', left_segloss, right_segloss)
			child_acc = fold.add('vectorAdder', left_acc, right_acc)
			node_segloss = fold.add('vectorAdder', child_segloss, node_segloss)
			acc = fold.add('vectorAdder', acc, child_acc) 
			return node_label_loss, node_segloss, acc

	input_data = jitter(tree.root.points)
	local_f = fold.add('pcEncoder', input_data)
	node_label_loss, node_segloss, acc = decode_node(tree.root, local_f)
	return node_label_loss, node_segloss, acc