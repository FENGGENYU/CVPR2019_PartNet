import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
from ._ext import grouping


class GroupPoints(Function):
    def forward(ctx, points, idx):
        b = points.size()[0]
        n = points.size()[1]
        c = points.size()[2]
        m = idx.size()[1]
        nsamples = idx.size()[2]
        out = torch.zeros(b, m, nsamples, c).cuda()
        ctx.save_for_backward(idx)
        ctx.b = b
        ctx.n = n
        ctx.c = c
        grouping.groupPoint_forward_cuda(b, n, c, m, nsamples, points, idx, out)
        return out

    def backward(ctx, grad_out):
        idx = ctx.saved_tensors[0]
        b = ctx.b
        n = ctx.n
        c = ctx.c
        m = idx.size()[1]
        nsamples = idx.size()[2]
        grad_points = torch.zeros(b, n, c).cuda()
        grouping.groupPoint_backward_cuda(b, n, c, m, nsamples, grad_out, idx, grad_points)
        return grad_points, None


class QueryBallPoint(Function):
    def __init__(self, radius, nsample):
        super(QueryBallPoint, self).__init__()
        self.radius = radius
        self.nsample = nsample
        #self.requires_grad = False

    def forward(ctx, xyz1, xyz2):
        b = xyz1.size()[0]
        n = xyz1.size()[1]
        m = xyz2.size()[1]
        idx = torch.IntTensor(b, m, ctx.nsample).cuda()
        pts_cnt = torch.IntTensor(b, m).cuda()
        grouping.queryBallPoint_cuda(b, n, m, ctx.radius, ctx.nsample, xyz1, xyz2, idx, pts_cnt)
        return idx, pts_cnt

    def backward(ctx, idx_grad, pts_grad):
        print('QueryBallPoint backward')
        return None, None


class GroupPointsModule(Module):
    def forward(self, points, idx):
        return GroupPoints(points, idx)

"""
def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value
    c = xyz1.get_shape()[2].value
    m = xyz2.get_shape()[1].value
    print b, n, c, m
    print xyz1, (b, 1, n, c)
    xyz1 = tf.tile(tf.reshape(xyz1, (b, 1, n, c)), [1, m, 1, 1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b, m, 1, c)), [1, 1, n, 1])
    dist = tf.reduce_sum((xyz1 - xyz2)**2, -1)
    print dist, k
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0, 0, 0], [-1, -1, k])
    val = tf.slice(out, [0, 0, 0], [-1, -1, k])
    print idx, val
    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx
"""
