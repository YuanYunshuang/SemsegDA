"""
Misc Utility functions
"""
import os
import logging
import datetime
import numpy as np

from collections import OrderedDict
import matplotlib.pylab as plt


def plot_grad_flow(named_parameters, fig):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n[-9])
            ave_grads.append(p.grad.abs().mean())

    plt.plot(ave_grads, color="b", linewidth=0.1)
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers", labelpad=20)
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.xticks(range(0,len(ave_grads)), layers, rotation='vertical')
    #plt.margins(0.2)
    fig.subplots_adjust(bottom=0.15)
    fig.canvas.draw()
    fig.show()
    plt.savefig('/home/robotics/pytorch-semseg/tmp.png', bbox_inches='tight')


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger("semseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
