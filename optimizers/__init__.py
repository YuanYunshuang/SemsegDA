import logging

from torch.optim import SGD, Adam

logger = logging.getLogger("ptsemseg")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
}


def get_optimizer_cls(cfg):
    if cfg["training"]["optimizer"] is None:
        logger.info("Using SGD optimizer")
        return SGD

    else:
        opt_name = cfg["training"]["optimizer"]["name"]
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        logger.info("Using {} optimizer".format(opt_name))
        return key2opt[opt_name]


def get_optimizer(train_parameters, cfg):
    name = cfg["training"]["optimizer"]["name"]
    opt_cfg = cfg["training"]["optimizer"]
    if name =="adam":
        return Adam(train_parameters, lr=opt_cfg["lr"], betas=(opt_cfg["betas"][0], opt_cfg["betas"][0]))
    elif name == "sdg":
        return Adam(train_parameters, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_dacay"])
    else:
        raise IOError("Invailid optimizer name: {}".format(name))

