import time
import os
from time import gmtime, strftime
from datetime import datetime
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
from torchfoldext import FoldExt
import util
from tensorboard_logger import configure, log_value

from dataloader import Data_Loader
import partnet as partnet_model

config = util.get_args()

config.cuda = not config.no_cuda
if config.gpu < 0 and config.cuda:
	config.gpu = 0
torch.cuda.set_device(config.gpu)
if config.cuda and torch.cuda.is_available():
	print("Using CUDA on GPU ", config.gpu)
else:
	print("Not using CUDA.")

SEED = 8701
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True

net = partnet_model.PARTNET(config)

if config.cuda:
	net = net.cuda() 

print("Loading data ...... ", end='\n', flush=True)
data_loader_batch = Data_Loader(config.data_path, config.training, config.split_num, config.total_num)

def my_collate(batch):
	a = torch.cat([x.shape for x in batch], 0)
	return batch, a

train_iter = torch.utils.data.DataLoader(
	data_loader_batch,
	batch_size=config.batch_size,
	shuffle=True,
	collate_fn=my_collate)
print("DONE")

opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.8)

print("Start training ...... ")

start = time.time()

net.train()

header = 'Time	   Epoch	 Iteration	  Progress(%)  LabelLoss  SegLoss	 Seg_acc(%)'
log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>11.2f},{:>10.2f},{:>10.2f}'.split(','))

if not os.path.exists(config.save_path):
	os.makedirs(config.save_path)
				
configure(config.save_path + "training_log/")
total_iter = config.epochs * len(train_iter)
step = 0
for epoch in range(config.epochs):
	scheduler.step()
	print(header)
	for batch_idx, batch in enumerate(train_iter):
		# compute points feature
		input_data = batch[1].cuda()
		jitter_input = torch.randn(input_data.size()).cuda()
		jitter_input = torch.clamp(0.01*jitter_input, min=-0.05, max=0.05)
		jitter_input += input_data
		points_feature = net.pointnet(jitter_input)
		# Split into a list of fold nodes per example
		enc_points_feature = torch.split(points_feature, 1, 0)
		# Initialize torchfold for *decoding*
		dec_fold = FoldExt(cuda=config.cuda)
		# Collect computation nodes recursively from decoding process
		dec_fold_nodes_label = []
		dec_fold_nodes_box = []
		dec_fold_nodes_acc = []
		for example, points_f in zip(batch[0], enc_points_feature):
			labelloss, boxloss, acc = partnet_model.decode_structure_fold(dec_fold, example, points_f)
			dec_fold_nodes_label.append(labelloss)
			dec_fold_nodes_box.append(boxloss)
			dec_fold_nodes_acc.append(acc)
		# Apply the computations on the decoder model
		dec_loss = dec_fold.apply(net, [dec_fold_nodes_label, dec_fold_nodes_box, dec_fold_nodes_acc])
		num_nodes = torch.cat([x.n_nodes for x in batch[0]], 0)
		label_loss = torch.mean(dec_loss[0]/num_nodes)
		seg_loss = torch.mean(dec_loss[1]/num_nodes)
		acc_mean = torch.mean(dec_loss[2]/num_nodes)
		acc = acc_mean.item()/2048.
		log_value('label_loss', label_loss.item(), step)
		log_value('seg_loss', seg_loss.item(), step)  
		log_value('acc', acc, step)
		total_loss = label_loss + seg_loss
		# Do parameter optimization
		opt.zero_grad()
		total_loss.backward()
		opt.step()
		# Report statistics
		if batch_idx % config.show_log_every == 0:
			print(
				log_template.format(
					strftime("%H:%M:%S", time.gmtime(time.time() - start)),
					epoch, config.epochs, 1 + batch_idx, len(train_iter),
					100. * (1 + batch_idx + len(train_iter) * epoch) /
					(len(train_iter) * config.epochs), label_loss.item(), seg_loss.item(), acc*100))
		step += 1
	if (epoch+1) % 100 ==0:
		#print("Saving temp models ...... ", flush=True)
		print("Saving models ...... ", end='', flush=True)
		torch.save(net.state_dict(), config.save_path + '/partnet_temp_%d.pkl'%epoch)

# Save the final models
print("Saving final models ...... ", end='', flush=True)
torch.save(net.state_dict(), config.save_path + '/partnet_final.pkl')
print("DONE")
