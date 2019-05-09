import os
from argparse import ArgumentParser

def get_args():
	parser = ArgumentParser(description='grass_pytorch')
	parser.add_argument('--part_code_size', type=int, default=128)
	parser.add_argument('--feature_size', type=int, default=128)
	parser.add_argument('--hidden_size', type=int, default=256)

	parser.add_argument('--epochs', type=int, default=1000)
	parser.add_argument('--batch_size', type=int, default=10)
	parser.add_argument('--show_log_every', type=int, default=1)
	parser.add_argument('--save_log', action='store_true', default=False)
	parser.add_argument('--save_log_every', type=int, default=3)
	parser.add_argument('--save_snapshot', action='store_true', default=False)
	parser.add_argument('--save_snapshot_every', type=int, default=5)
	parser.add_argument('--no_plot', action='store_true', default=True)
	parser.add_argument('--lr', type=float, default=.001)
	parser.add_argument('--lr_decay_by', type=float, default=1)
	parser.add_argument('--lr_decay_every', type=float, default=1)
	parser.add_argument('--no_cuda', action='store_true', default=False)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--data_path', type=str, default='./data/airplane/')
	parser.add_argument('--save_path', type=str, default='./models/airplane/')
	parser.add_argument('--output_path', type=str, default='./results/airplane/')
	#training_split_num airplane:510 sofa:510 bike:125 helico:80 table:450 chair:800
	#total_num          airplane:630 sofa:630 bike:155 helico:100 table:583 chair:999
	parser.add_argument('--split_num', type=int, default=510)
	parser.add_argument('--total_num', type=int, default=630)
	parser.add_argument('--training', type=bool, default=False)
	#label_category airplane:4 sofa:4 bike:8 helico:5 table:4
	parser.add_argument('--label_category', type=int, default=4)
	parser.add_argument('--resume_snapshot', type=str, default='')
	args = parser.parse_args()
	return args
