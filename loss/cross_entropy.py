import torch
import torch.nn.functional as F
import numpy as np


def cross_entropy2d_weighted(input, target, weight=None):
    n, c, h, w = input.size()
    nt, ct, ht, wt = target.size()

    if weight is None:
        n_l = torch.Tensor([target.numel()]).to(dtype=torch.float).cuda()
        l0 = n_l / (target == 0).sum().to(dtype=torch.float)
        l1 = n_l / (target == 1).sum().to(dtype=torch.float)
        l2 = n_l / (target == 2).sum().to(dtype=torch.float)

        weight = torch.cat((l0, l1, l2))
        if torch.isinf(l0) or torch.isinf(l1) or torch.isinf(l2):
            weight[torch.isinf(weight)] = 0
            weight = weight / weight.sum()
        else:
            weight = weight / weight.sum()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, ignore_index=255
    )
    return loss


def cross_entropy2d(input, target, weight=None):
    if isinstance(input, torch.Tensor):
        n, c, h, w = input.size()
        nt, ct, ht, wt = target.size()
    else:
        n, c, h, w = input.shape
        nt, ct, ht, wt = target.shape

    if weight is None:
        n_l = torch.Tensor([target.numel()]).to(dtype=torch.float).cuda()
        ws = torch.zeros(input.size()[1]).to(dtype=torch.float).cuda()
        for i in range(input.size()[1]):
            ws[i] = n_l / (target == i).sum().to(dtype=torch.float)
        # if np.random.uniform() > 0.7:
        #    weight[0] = 0
        if torch.any(torch.isinf(ws)):
            ws[torch.isinf(ws)] = 0
            ws = ws / ws.sum()
        else:
            ws = ws / ws.sum()
    # print(torch.unique(input))
    # print(torch.unique(target))
    # print(ws)
    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c).to(target.device)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=ws,  ignore_index=255
    )
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            target.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
