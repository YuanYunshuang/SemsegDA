# FCN Small

import functools
import torch
import torch.nn as nn
from loss import cross_entropy2d
from torchvision.models import resnet50


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class AAN(nn.Module):
    def __init__(self,
                 n_classes=2,
                 input_size=(512, 512),
                 ):
        super(AAN, self).__init__()
        self.resnet = resnet50(pretrained=True, progress=False)
        self.outputsF = [Hook(layer[1]) for layer in list(self.resnet._modules.items())[:8]] # maxpool, layer1-4
        self.outputsB = [Hook(layer[1], backward=True) for layer in list(self.resnet._modules.items())[:8]]


    def forward(self, x):
        out = self.resnet(x)

        return out

class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

