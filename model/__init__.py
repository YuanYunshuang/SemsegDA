import copy
import torchvision.models as model
import torch

from model.fcn import FCN
from model.drcn import DRCN
from model.unet import Unet


def get_model(cfg):
    model_dict = cfg["model"]
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if name == "fcn":
        model = model(n_classes=cfg["data"]["n_classes"])
        model.apply(weights_init)
    elif name == "drcn":
        model = model(cfg)
        model.apply(weights_init)
    elif name == "unet":
        model = model(cfg)
        model.apply(weights_init)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn": FCN,
            "drcn": DRCN,
            "unet": Unet
        }[name]

    except:
        raise ("Model {} not available".format(name))


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def set_trainable(model, mode=None):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if mode is None:
        return model
    elif mode=='icnet':
        for p in model.named_parameters():
            if 'classification' in p[0] \
                 or 'cff' in p[0]:
                 # or 'cff' in p[0] \
                 # or 'cff' in p[0] \
                 # or 'cff' in p[0] \
                p[1].requires_grad = False
