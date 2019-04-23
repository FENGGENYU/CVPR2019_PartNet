import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
from ._ext import interpolate


class InterpolateFunction(Function):
    def forward(ctx, points, idx, weight):
        # points: (b, c, m)
        # idx: (b, n, 3)
        # weight: (b, n, 3)
        b = points.size(0)
        c = points.size(1)
        m = points.size(2)
        n = idx.size(1)
        out = torch.zeros(b, c, n).cuda()
        interpolate.three_interpolate_wrapper(b, c, n, m, points, idx, weight, out)

        ctx.b = b
        ctx.c = c
        ctx.n = n
        ctx.m = m
        ctx.save_for_backward(idx, weight)
        return out

    def backward(ctx, out_grad):
        points_grad = torch.zeros(ctx.b, ctx.c, ctx.m).cuda()
        idx, weight = ctx.saved_tensors
        interpolate.three_interpolate_grad_wrapper(ctx.b, ctx.c, ctx.n, ctx.m, out_grad, idx, weight, points_grad)
        return points_grad, None, None
