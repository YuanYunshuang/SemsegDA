import torch
import torch.nn.functional as F


def focal_loss2d(input, target, gamma=1, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ct, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1,1)
    posteri = torch.gather(input, 1, target).squeeze()
    if weight is None:
        ws = torch.zeros(input.size()[1])
        ps = torch.zeros(input.size()[1])
        for i in range(input.size()[1]):
            p_i = posteri[target.squeeze()==i]
            if len(p_i) > 0:
                ws[i] = len(posteri) / len(p_i)
            else:
                ws[i] = 0
            ps[i] = -torch.sum((1-p_i)**gamma * torch.log(p_i))
    ws = ws / ws.sum()
    loss = (ws * ps).sum() 

    return loss


