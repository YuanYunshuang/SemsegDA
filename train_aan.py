import torch
import numpy as np
import random
import argparse
import yaml

from loader import get_loader
from model.aan import AAN
from loss import dist




def train(cfg):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    loader_src = get_loader(cfg, "train", pathname="path_source")
    loader_tgt = get_loader(cfg, "train", pathname="path_target")

    # Setup model
    aan_src = AAN()
    aan_tgt = AAN()
    aan_transfer = AAN()

    # Train
    ws = [1,2,4]
    wt = [4, 2]
    alpha = 1e-4
    for img_src, _ in loader_src:
        # generate input noise image
        img_src = img_src[:,:3,:,:]
        noise = torch.randn_like(img_src)
        out_noise = aan_transfer(noise)
        out_src = aan_src(img_src)
        content_features = []
        noise_features = []
        for i in range(5,8):
            content_features.append(aan_src.outputsF[i].output)
            noise_features.append(aan_transfer.outputsF[i].output)
        loss_content = dist.content_loss(content_features, noise_features, weights=ws)
        conv1_sum = 0 # gram matrices sum
        conv2_sum = 0
        for img_tgt, _ in loader_tgt:
            img_tgt = img_tgt[:,:3,:,:]
            out_tgt = aan_tgt(img_tgt)
            conv1 = dist.GramMatrix(aan_src.outputsF[3].output)
            conv2 = dist.GramMatrix(aan_src.outputsF[4].output)
            conv1_sum = conv1_sum + conv1
            conv2_sum = conv2_sum + conv2

        con1_mean = conv1_sum / len(loader_tgt)
        con2_mean = conv2_sum / len(loader_tgt)
        loss_style = dist.style_loss(noise_features, [con1_mean, con2_mean], weights=wt)
        
        loss = loss_content + alpha * loss_style
        loss.backward()
        noise = aan_transfer.outputsB.input[0]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)

    train(cfg)
