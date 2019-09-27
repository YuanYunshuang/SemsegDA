import os
import torch
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow

from torch.utils import data
from torchvision import transforms
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


class threeCityLoader(data.Dataset):
    """3 cities dataset Loader
    """
    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(320, 320),
        channels=4,
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
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = len(label_map)
        self.n_channels = channels
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size, channels)
        self.files = {}

        self.images_base = os.path.join(self.root, self.split,"images",)
        self.annotations_base = os.path.join(self.root, self.split, "labels")

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")[:200]

        self.void_classes = []
        self.valid_classes = [0, 1, 2]
        self.class_names = label_map.keys()

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, label_map.values()))
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

        lbl = imread(lbl_path)
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
        msi = img[:,:,:3]
        dsm = np.expand_dims(img[:,:,-1], 0)
        msi = msi.astype(np.float64)
        # if self.img_norm:
        #     dsm = (dsm - np.mean(dsm)) / np.std(dsm - np.mean(dsm))
        #     for i in range(3):
        #         msi[:,:,i] = (msi[:,:,i] - np.mean(msi[:,:,i])) / np.std(msi[:,:,i])
        #imsave('/home/robotics/image.png', msi)
        # NHWC -> NCHW
        msi = msi.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        lbl = np.expand_dims(lbl, 0)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        # imshow(img.transpose(1,2,0)[:,:,:-1])
        # imshow(lbl.squeeze()*80)
        msi = torch.from_numpy(msi).float()
        dsm = torch.from_numpy(dsm).float()
        lbl = torch.from_numpy(lbl).long()

        if self.img_norm:
            norm = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
            dsm = norm(dsm)
            norm = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            msi = norm(msi)

        input = torch.cat((msi, dsm), 0)

        return input, lbl

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


if __name__ == "__main__":
    img = imread('/home/robotics/rssrai2019/data_preprocessed1/val/images/0/20160421_L1A0001537716_55.png')
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/home/robotics/ma_thesis_data/lgln_3city/dataset/C1_20cm"
    dst = threeCityLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        import pdb

        pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
