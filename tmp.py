import os
import torch
import random
import numpy as np
import argparse
import yaml
import tqdm
from PIL import Image
from model import Unet
import utils
from utils.metrics import averageMeter, runningScore
from utils.visualizer import Visualizer
from configs.base_options import BaseOptions
from loader import get_loader
from utils.batch_statistics import StatsRecorder


def eval(cfg):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup evaluation data
    loader_source = get_loader(cfg, "train")
    # data_eval_labels = utils.recursive_glob(os.path.join(cfg["data"]["path"], 'labels'))

    # Setup model
    model = Unet(cfg).to(device)
    checkpoint = torch.load(cfg["training"]["checkpoint"])
    model.load_state_dict(checkpoint["model_state"])
    stats = None
    model.eval()
    for images, labels in tqdm.tqdm(loader_source):
        model.set_input(images, labels)
        model.forward()
        if stats is None:
            stats = [StatsRecorder() for i in range(len(model.hooks))]
        for i, hook in enumerate(model.hooks):
            activation = hook.output
            b, c, h, w = activation.shape
            activation = activation.transpose(0, 1).reshape(c, -1).transpose(0, 1)
            stats[i].update(activation.cpu().data.numpy())

    print([s.mean for s in stats])
    print([s.std for s in stats])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="/home/robotics/SemsegDA/configs/adapt_unet_isprs.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)

    eval(cfg)