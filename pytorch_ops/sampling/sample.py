import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
from ._ext import farthestpointsampling as fps


class SampleFunction(Function):
    def __init__(self, npoints):
        self.npoints = npoints
    def forward(ctx, inp):
        b = inp.size()[0]
        n = inp.size()[1]
        temp = torch.zeros(32, n).cuda()
        idx = torch.IntTensor(b, ctx.npoints).zero_().cuda()
        out = torch.zeros(b, ctx.npoints, 3).cuda()
        fps.farthestpointsampling_forward_cuda(b, n, ctx.npoints, inp, temp, idx)
        fps.gatherpoint_forward_cuda(b, n, ctx.npoints, inp, idx, out)
        ctx.save_for_backward(idx)
        ctx.b = b
        ctx.n = n
        return out, idx

    def backward(ctx, out_grad, idx_grad):
        idx  = ctx.saved_tensors[0]
        b = ctx.b
        n = ctx.n
        m = ctx.npoints
        inp_grad = torch.zeros(b, n, 3).cuda()
        fps.gatherpoint_backward_cuda(b, n, m, out_grad, idx, inp_grad)
        return inp_grad, None


class FarthestSample(Module):
    def __init__(self, npoints):
        super(FarthestSample, self).__init__()
        self.npoints = npoints
    def forward(self, inp):
        return SampleFunction(self.npoints)(inp)
