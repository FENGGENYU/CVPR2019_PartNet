import torchfold
from torchfold import Fold
import torch
from torch.autograd import Variable


class FoldExt(Fold):

    def __init__(self, volatile=False, cuda=False):
        Fold.__init__(self, volatile, cuda)


    def add(self, op, *args):
        """Add op to the fold."""
        self.total_nodes += 1
        if not all([isinstance(arg, (
            Fold.Node, int, torch.Tensor, torch.FloatTensor, torch.LongTensor, Variable)) for arg in args]):
            raise ValueError(
                "All args should be Tensor, Variable, int or Node, got: %s" % str(args))
        if args not in self.cached_nodes[op]:
            step = max([0] + [arg.step + 1 for arg in args
                              if isinstance(arg, Fold.Node)])
            node = Fold.Node(op, step, len(self.steps[step][op]), *args)
            self.steps[step][op].append(args)
            self.cached_nodes[op][args] = node
        return self.cached_nodes[op][args]


    def _batch_args(self, arg_lists, values):
        res = []
        for arg in arg_lists:
            r = []
            if isinstance(arg[0], Fold.Node):
                if arg[0].batch:
                    for x in arg:
                        r.append(x.get(values))
                    res.append(torch.cat(r, 0))
                else:
                    for i in range(2, len(arg)):
                        if arg[i] != arg[0]:
                            raise ValueError("Can not use more then one of nobatch argument, got: %s." % str(arg))
                    x = arg[0]
                    res.append(x.get(values))
            else:
                # Below is what this extension changes against the original version:
                #   We make Fold handle float tensor
                try:
                    if (isinstance(arg[0], Variable)):
                        var = torch.cat(arg, 0)
                    else:
                        var = Variable(torch.cat(arg, 0), volatile=self.volatile)
                    if self._cuda:
                        var = var.cuda()
                    res.append(var)
                except:
                    print("Constructing float tensor from %s" % str(arg))
                    raise
        return res