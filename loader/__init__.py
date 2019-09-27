from torch.utils import data

from loader.dsm_loader import dsmLoader
from loader.three_city_loader import threeCityLoader
from augmentations import get_composed_augmentations


def get_loader_cls(name):
    """get_loader

    :param name:
    """
    return {
        "dsm": dsmLoader,
        "threeCity": threeCityLoader
    }[name]


def get_loader(cfg, setname, pathname='path', sampler=None):
    # Setup dataset name and path
    data_loader = get_loader_cls(cfg["data"]["dataset"])
    assert pathname=='path_source' or pathname=='path_target' or pathname=='path', "Pathname not correct!"
    data_path = cfg["data"][pathname]

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    if setname == 'train':

        t_loader = data_loader(
            data_path,
            is_transform=True,
            split=cfg["data"]["train_split"],
            img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
            channels=cfg["data"]["channels"],
            augmentations=data_aug,
            suffix=cfg["data"]["suffix"]
        )
        if sampler is not None:
            loader = data.DataLoader(
                t_loader,
                batch_size=cfg["training"]["batch_size"],
                sampler=sampler(t_loader),
                num_workers=cfg["training"]["n_workers"],
            )
        else:
            loader = data.DataLoader(
                t_loader,
                batch_size=cfg["training"]["batch_size"],
                shuffle=True,
                num_workers=cfg["training"]["n_workers"],
            )

    elif setname == 'val':

        v_loader = data_loader(
            data_path,
            is_transform=True,
            split=cfg["data"]["val_split"],
            img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
            channels=cfg["data"]["channels"],
            suffix=cfg["data"]["suffix"]
        )

        loader = data.DataLoader(
            v_loader,
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"]["n_workers"]
        )

    else:
        raise TypeError("Setname: {} is not supported, please choose 'train' or 'val' as setname".format(setname))

    return loader


def get_sampler_cls(name):
    return {

    }[name]


def get_sampler(cfg):
    name = cfg['training']['sampler']
    if name is None:
        return None
    return get_sampler_cls(name)
