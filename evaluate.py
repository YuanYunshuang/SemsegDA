import os
import torch
import random
import numpy as np
import argparse
import yaml
import tqdm
from PIL import Image
from model import DRCN
import utils
from utils.metrics import averageMeter, runningScore
from utils.visualizer import Visualizer
from configs.base_options import BaseOptions
from loader.three_city_loader import threeCityLoader


def eval(cfg):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup evaluation data
    data_eval_images = utils.recursive_glob(os.path.join(cfg["data"]["path"], 'images'))
    # data_eval_labels = utils.recursive_glob(os.path.join(cfg["data"]["path"], 'labels'))

    # Setup model
    model = DRCN(cfg).to(device)
    checkpoint = torch.load(cfg["training"]["checkpoint"])
    model.load_state_dict(checkpoint["model_state"])

    # Setup Metrics and visualizer
    running_metrics_val = runningScore(cfg["data"]["n_classes"])

    # Start training
    utils.mkdirs(cfg["training"]["checkpoint"])

    s = cfg["data"]["img_rows"]
    for img_name in tqdm.tqdm(data_eval_images):
        img = np.array(Image.open(img_name))
        lbl = np.array(Image.open(img_name.replace('images', 'labels')))
        w, h, _ = img.shape
        out = np.zeros((6,w,h))
        for x in range(0, w - s, 200):
            for y in range(0, h - s, 200):
                img_input, lbl_input = threeCityLoader.transform(img[x:x+s,y:y+s,:], lbl[x:x+s,y:y+s])
                model.set_input(img_input.unsqueeze(0), lbl_input.unsqueeze(0))
                model.inference()
                out[:, x: x + s, y: y + s] += model.out1.cpu().detach().numpy().squeeze()
        max_x = (w - s) // 200 * 200
        max_y = (h - s) // 200 * 200
        pred = out[:, :max_x, :max_y]
        pred = pred.argmax(0).squeeze()
        gt = lbl[:max_x, :max_y]

        running_metrics_val.update(gt, pred)

        score, class_iou = running_metrics_val.get_scores()
        for k, v in score.items():
            print(k, v)

        for k, v in class_iou.items():
            print("{}: {}".format(k, v))

    running_metrics_val.reset()



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

    eval(cfg)
