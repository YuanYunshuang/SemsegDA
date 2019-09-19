import torch
import torch.multiprocessing as mp
import numpy as np

def coral_loss2d(source, target):
    ns, nt = source.size(0), target.size(0)
    d = source.size(1)
    source = source.view(d, -1).transpose(0, 1) # .transpose(1,2).transpose(0,1)
    target = target.view(d, -1).transpose(0, 1) # .transpose(1,2).transpose(0,1)
    # loss = 0.0
    # for s, t in list(zip(source, target)):
    loss = coral_loss(source, target)

    return loss


def coral_loss(source, target):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ns, nt = source.size(0), target.size(0)
    d = source.size(1)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(DEVICE) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    # loss = loss / (4 * d * d)

    return loss


def weights_loss(model_src, model_tgt):
    weights_sum = 0.0
    n_weights = 0
    for m, n in zip(model_src.modules(), model_tgt.modules()):

        if isinstance(m, torch.nn.Conv2d) and isinstance(n, torch.nn.Conv2d):
            weights_sum = weights_sum + torch.sum(torch.abs(m.weight - n.weight))
            n_weights = n_weights + m.weight.numel()

    return weights_sum / n_weights
