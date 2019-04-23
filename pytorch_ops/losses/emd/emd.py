import torch
from torch.autograd import Function,Variable
from torch.nn.modules.module import Module
from .._ext import emd


class EMDFunction(Function):
    def forward(ctx, xyz1, xyz2):
        b = xyz1.size()[0]
        n = xyz1.size()[1]
        m = xyz2.size()[1]
        ctx.b=b
        ctx.n=n
        ctx.m=m
        match = torch.zeros(b, m, n).cuda()
        temp = torch.zeros(b, (n + m) * 2).cuda()
        cost = torch.zeros(b).cuda()
        emd.approxmatch_cuda_forward(xyz1, xyz2, match, temp)
        ctx.save_for_backward(xyz1, xyz2)
        ctx.match = match
        emd.matchcost_cuda_forward(xyz1, xyz2, match, cost)
        return cost

    def backward(ctx,grad_cost):
        b = ctx.b
        n = ctx.n
        m = ctx.m
        xyz1, xyz2 = ctx.saved_tensors
        grad1 = torch.zeros(b, n, 3).cuda()
        grad2 = torch.zeros(b, m, 3).cuda()
        emd.matchcost_cuda_backward(xyz1, xyz2, ctx.match, grad1, grad2)
        grad_cost=torch.unsqueeze(torch.unsqueeze(grad_cost,1),2)
        grad1 = grad1*grad_cost
        grad2 = grad2*grad_cost
        return grad1, grad2


class EMDModule(Module):
    def forward(self, input1, input2):
        return EMDFunction()(input1, input2)
