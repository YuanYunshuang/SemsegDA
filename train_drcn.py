import os
import torch
import random
import numpy as np
import argparse
import yaml
from loader import get_loader
from model import DRCN
import utils
from utils.metrics import averageMeter, runningScore
from utils.visualizer import Visualizer
from configs.base_options import BaseOptions

def train(cfg):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    loader = get_loader(cfg, "train")

    # Setup model
    model = DRCN(cfg).to(device)

    # Setup Metrics and visualizer
    running_metrics_val = runningScore(cfg["data"]["n_classes"])
    opt = BaseOptions()
    visualizer = Visualizer(opt)

    # Start training
    utils.mkdirs(cfg["training"]["checkpoint"])

    best_iou = -100.0
    total_iters = 0
    epoch = 0
    train_epochs = cfg["training"]["epochs"]
    while epoch < train_epochs:
        epoch += 1
        visualizer.reset()
        for images, labels in loader:
            total_iters += cfg["training"]["batch_size"]
            model.set_input(images, labels)
            model.optimize_parameters()

            if total_iters % cfg["training"]["print_interval"]==0:
                print_info = "Epoch:[{:4d}/{:4d}] Iter: [{:6d}] loss1: {:.5f}  loss2: {:.5f}  lr: {:.8f}"\
                    .format(epoch, train_epochs, total_iters, model.loss1.item(), model.loss2.item(), model.optimizer.defaults['lr'])
                print(print_info)

        if epoch % opt.visual.display_freq == 0:
            #visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            losses = {'loss1': model.loss1.item(),
                      'loss2': model.loss2.item(),
                      'total_loss': model.loss1.item() + model.loss2.item()}
            visualizer.plot_current_losses(epoch, epoch / train_epochs, losses)

        if epoch % cfg["training"]["val_interval"]==0:
            for images, labels in loader:
                model.set_input(images, labels)
                model.inference()
                preds = torch.argmax(model.out1, 1).cpu().numpy()
                labels = labels.data.numpy().squeeze()

                running_metrics_val.update(labels, preds)

            score, class_iou = running_metrics_val.get_scores()
            for k, v in score.items():
                print(k, v)

            for k, v in class_iou.items():
                print("{}: {}".format(k, v))

            running_metrics_val.reset()

            if score["Mean IoU : \t"] >= best_iou:
                best_iou = score["Mean IoU : \t"]
                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": model.optimizer.state_dict(),
                    "scheduler_state": model.scheduler.state_dict(),
                    "best_iou": best_iou,
                }
                save_path = os.path.join(
                    cfg["training"]["checkpoint"],
                    "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                )
                torch.save(state, save_path)




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
