import os
import torch
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow

from torch.utils import data

from utils import recursive_glob
from augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale

label_map = {
    'Impervious_surfaces': [255, 255, 255],
    'Building': [0, 0, 255],
    'Low_vegetation': [0, 255, 255],
    'Tree': [0, 255, 0],
    'Car': [255, 255, 0],
    'Clutter': [255, 0, 0],
}



class dsmLoader(data.Dataset):
    """3 cities dataset Loader
    """
    def __init__(
        self,
        root,
        split="train",
        index=1, # class for binary classification
        is_transform=False,
        img_size=(320, 320),
        channels=1,
        augmentations=None,
        img_norm=True,
        test_mode=False,
        suffix=".png"
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.index = index
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = len(label_map)
        self.n_channels = channels
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size, channels)
        self.files = {}

        self.images_base = os.path.join(self.root, self.split,"images",)
        self.annotations_base = os.path.join(self.root, self.split, "labels")

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = np.arange(len(label_map))
        self.void_classes = self.void_classes[self.void_classes != self.index]
        self.valid_classes = [self.index]
        self.class_names = label_map.keys()

        self.ignore_index = 255
        self.label_colours = dict(zip(range(len(label_map)), list(label_map.values())))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images in dataset %s" % (len(self.files[split]), split, root.split('/')[-1]))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            os.path.basename(img_path)
        )
        img = imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = imread(lbl_path) + 1
        lbl[lbl!=self.index] = 0
        lbl[lbl==self.index] = 1
        #imshow(img[:, :, :3])
        #imshow(img[:, :, 3]*10)
        #imshow(lbl*80)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        dsm = img.astype(np.float64)
        if self.img_norm:
            dsm = (dsm - np.mean(dsm)) / np.std(dsm - np.mean(dsm))

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        dsm = np.expand_dims(dsm, 0)
        lbl = np.expand_dims(lbl, 0)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")
        classes = np.unique(lbl)
        for i in range(len(classes)):
            if not classes[i] in [1., 0.]:
                raise ValueError("Segmentation map contained invalid class values")
        # imshow(img.transpose(1,2,0)[:,:,:-1])
        # imshow(lbl.squeeze()*80)
        dsm = torch.from_numpy(dsm).float()
        lbl = torch.from_numpy(lbl).long()

        return dsm, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, lbl):
        # Put all void classes to zero
        mask = np.ones(lbl.shape[:2], dtype=np.uint8) * self.ignore_index
        for _validc in self.valid_classes:
            clr = self.class_map[_validc]
            mask_bool = np.array(lbl == clr).all(axis=2)
            mask[mask_bool] = _validc
        return mask

