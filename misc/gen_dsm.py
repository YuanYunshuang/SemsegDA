import tqdm
color_map = {
    'Impervious_surfaces': [255, 255, 255],
    'Building': [0, 0, 255],
    'Low_vegetation': [0, 255, 255],
    'Tree': [0, 255, 0],
    'Car': [255, 255, 0],
    'Clutter': [255, 0, 0],
}

import os
import numpy as np
from scipy.misc import imread, imsave, imshow

path = '/home/robotics/ma_thesis_data/isprs/preprocessed'
save = '/home/robotics/ma_thesis_data/isprs/dataset512_dsm'
cities = ['Potsdam8cm', 'Vaihingen8cm']
size = 800


def gen_images():

    for city in cities:
        data_path = os.path.join(path, city)
        print(city)
        for patch in tqdm.tqdm(range(1, 38)):
            dsm = imread(os.path.join(data_path,  str(patch), 'DSM.tif'))
            label = imread(os.path.join(data_path,  str(patch), 'L.png'))
            row, col = label.shape
            if len(dsm.shape)==3:  # some dsm images are with 3 channels of same values
                dsm = dsm[:,:,0]
            if patch % 2 == 0:
                img_path = os.path.join(save, city, 'train', 'images') # last channel is dsm
                label_path = os.path.join(save, city, 'train', 'labels')
            else:
                if patch < 8:
                    img_path = os.path.join(save, city, 'val', 'images') # last channel is dsm
                    label_path = os.path.join(save, city, 'val', 'labels')
                else:
                    img_path = os.path.join(save, city, 'test', 'images') # last channel is dsm
                    label_path = os.path.join(save, city, 'test', 'labels')
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            if not os.path.exists(label_path):
                os.makedirs(label_path)
            if 'test' in label_path:
                imsave(os.path.join(img_path, str(patch)) + '.png', dsm)
                imsave(os.path.join(label_path, str(patch)) + '.png', label)
            elif 'val' or 'train' in label_path:
                for i in range(300):
                    x = np.random.randint(0, row - size)
                    y = np.random.randint(0, col - size)
                    I = dsm[x:x+size,y:y+size]
                    lbl = label[x:x+size,y:y+size]
                    #imshow(I[:,:,:-1])
                    #imshow(lbl*80)
                    imsave(os.path.join(img_path, str(patch) +'_'+str(i)) +'.png', I)
                    imsave(os.path.join(label_path, str(patch) +'_'+str(i)) +'.png', lbl)




gen_images()