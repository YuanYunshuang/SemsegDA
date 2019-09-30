import os
import torch
import random
import numpy as np
import argparse
import yaml, shutil
from loader import get_loader
from model import get_model
import utils
from utils.metrics import averageMeter, runningScore
from utils.visualizer import Visualizer
from configs.base_options import BaseOptions

AccNames = ['OA', 'Impervious_surfaces', 'Building', 'Low_vegetation', 'Tree', 'Car', 'Clutter']


def train(cfg, logger):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    loader_train = get_loader(cfg, "train")
    loader_val = get_loader(cfg, "val")

    # Setup model
    model = get_model(cfg).to(device)
    start_epoch = 1
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            print(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            start_epoch = checkpoint["epoch"]
            del checkpoint

        else:
            print("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    # Setup Metrics and visualizer
    running_metrics_val = runningScore(cfg["data"]["n_classes"])
    val_loss_meter = averageMeter()
    opt = BaseOptions()
    visualizer = Visualizer(opt)

    # Start training
    utils.mkdirs(cfg["training"]["checkpoint"])

    best_iou = -100.0
    epoch = start_epoch
    train_epochs = cfg["training"]["epochs"]
    iters_per_epoch = len(loader_train)
    while epoch < train_epochs:
        visualizer.reset()
        for iter, (images, labels) in enumerate(loader_train):
            model.set_input(images, labels)
            model.optimize_parameters()

            if iter % cfg["training"]["print_interval"]==0 and iter!=0:
                print_info = "Epoch:[{:2d}/{:2d}] Iter: [{:4d}/{:4d}] loss: {:.5f}  lr: {:.5f}"\
                    .format(epoch, train_epochs, iter, iters_per_epoch, model.loss.item(), model.optimizer.defaults['lr'])
                print(print_info)

            if iter % cfg["training"]["val_interval"] == 0 and iter!=0:
                for images, labels in loader_val:
                    model.set_input(images, labels)
                    model.inference()
                    preds = torch.argmax(model.out, 1).cpu().numpy()
                    labels = labels.data.numpy().squeeze()

                    running_metrics_val.update(labels, preds)
                    val_loss_meter.update(model.loss.item())

                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                losses = {'loss': val_loss_meter.avg, 'zero': 0}
                score, class_iou = running_metrics_val.get_scores()
                accs = []
                accs.append(score["Overall Acc: \t"])
                accs.extend(list(class_iou.values()))
                accs = dict(zip(AccNames, accs))
                tmp = iter/iters_per_epoch
                visualizer.plot_current_losses(epoch, tmp, losses)
                visualizer.plot_current_accuracy(epoch, tmp, accs)
                logger.info("Epoch:{:03d} val_loss:{:.05f}"
                            .format(epoch, val_loss_meter.avg))
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))

                for k, v in class_iou.items():
                    print("{}: {}".format(k, v))
                    logger.info("{}: {}".format(k, v))

                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": model.optimizer.state_dict(),
                        "scheduler_state": model.scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        cfg["training"]["checkpoint"],
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["name"]),
                    )
                    torch.save(state, save_path)
        epoch += 1


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

    logdir = cfg["training"]["checkpoint"]
    logger = utils.get_logger(logdir)
    shutil.copy(args.config, logdir)

    train(cfg, logger)
