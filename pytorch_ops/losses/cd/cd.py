import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
from .._ext import cd


class CDFunction(Function):
    def forward(ctx, xyz1, xyz2):
        b = xyz1.size()[0]
        n = xyz1.size()[1]
        m = xyz2.size()[1]
        ctx.b=b
        ctx.n=n
        ctx.m=m
        dist1 = torch.zeros(b, n).cuda()
        dist2 = torch.zeros(b, m).cuda()
        idx1 = torch.IntTensor(b, n).zero_().cuda()
        idx2 = torch.IntTensor(b, m).zero_().cuda()
        cd.cd_forward_cuda(b, n, xyz1, m, xyz2, dist1, idx1, dist2, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, idx1, dist2, idx2

    def backward(ctx, grad_dist1, grad_idx1, grad_dist2, grad_idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        b = ctx.b
        n = ctx.n
        m = ctx.m
        grad_xyz1 = torch.zeros(b, n, 3).cuda()
        grad_xyz2 = torch.zeros(b, m, 3).cuda()
        cd.cd_backward_cuda(b, n, xyz1, m, xyz2, grad_dist1, idx1, grad_dist2,
                            idx2, grad_xyz1, grad_xyz2)
        return grad_xyz1, grad_xyz2


class CDModule(Module):
    def forward(self, input1, input2):
        return CDFunction()(input1, input2)
