import torch
import torch.nn.functional as F


def LD(rs, rt):
    loss = -torch.sum(torch.log(rs)) - torch.sum(torch.log(1 - rt))

    return loss

def LG(rt, encoder_s, encoder_t, la=100):
    weights_sum = 0
    n_weights = 0
    for m, n in zip(encoder_s.modules(), encoder_t.modules()):
        if isinstance(m, torch.nn.Conv2d) and isinstance(n, torch.nn.Conv2d):
            weights_sum = weights_sum + torch.sum(torch.abs(m.weight - n.weight))
            n_weights = n_weights + m.weight.numel()
    # tmp = weights_sum / n_weights
    # print(tmp.data * 10000)
    loss = -torch.sum(torch.log(rt)) + la * weights_sum / n_weights

    return loss

