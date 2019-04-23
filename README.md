# PartNet
Code for PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation

### Intruduction

We opt for top-down recursive decomposition and develop the first deep learning model for hierarchical segmentation of 3D shapes, based on recursive neural networks. Starting from a full shape represented as a point cloud, our model performs recursive binary decomposition, where the decomposition network at all nodes in the hierarchy share weights. At each node, a node classifier is trained to determine the type (adjacency or symmetry) and stopping criteria of its decomposition.

### Dependencies

Requirements:
- Python 3.5 with numpy, scipy, torchfold, tensorboard and etc.
- [PyTorch](https://pytorch.org/resources)

Our code has been tested with Python 3.5, PyTorch 0.4.0, CUDA 8.0 on Ubuntu 16.04.

## Datasets and Pre-trained weights
The input pointcloud and training hierarchical trees are on [Here](https://www.dropbox.com/sh/7nuqb9wphsjkzko/AAAgy8zzmeRFsNuGuYCxUUWTa?dl=0).
Each category contains following folds:
- models_2048_points_normals: normalized input shape pointcloud with normal
- training_data_models_segment_2048_normals: GT part pointcloud with normal
- training_trees: hierarchical structure(ops.mat) with symmetric parameters(syms.mat). The labels(labels.mat) are for testing.

The Pre-trained weights are on [Here](https://www.dropbox.com/sh/um1li37bnbkpuck/AAAaCAuXWaY050E7W5b42XT1a?dl=0).

### Usage: Demo
Require 3GB RAM on the GPU and 5sec to run.
This script takes as input a normalized 2048*6 pointcloud (Sampled from ShapeNet). Please download Pre-trained weights of airplane first.
```
python test_demo.py
```
![input](./picture/airplane.png)

### Usage: Training

Put data of each category in ./data/category_name(eg ./data/airplane) 

Build extention for each op in ./pytorch_ops using build.py
```
python build.py
```
Then run traning process
```
python train.py
```

More training arguments are set in util.py
```
'--epochs' (number of epochs; default=1000)
'--batch_size' (batch size; default=10)
'--show_log_every' (show training log for every X frames; default=3)
'--no_cuda' (don't use cuda)
'--gpu' (device id of GPU to run cuda)
'--data_path' (dataset path, default='data')
'--save_path' (trained model path, default='models')
'--output_path' (segmented result path, default='results')
'--training' (training or testing, default=False)
'--split_num' (training data size for each category)
'--total_num' (full data size for each category, only for testing)
'--label_category' (semantic labels for each category, only for testing)
```
### Usage: Testing

We measure AP(%) with IoU threshold being 0:25 and 0:5, respectively.
```
python ap_evaluate.py
```
Segmentation results and its corresponding GT can also be found in ./results/category_name(eg ./data/airplane).

PS: More codes for results showing symmetric parameters is coming on the way.
PPS: If you want to try more new shapes, please make sure that them are oriented and normalized as our shape.

## Citation
If you use this code, please cite the following paper.
```
@inproceedings{yu2019partnet,
    title = {PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation},
    author = {Fenggen Yu and Kun Liu and Yan Zhang and Chenyang Zhu and Kai Xu},
    booktitle = {CVPR},
    pages = {to appear},
    year = {2019}
}
```