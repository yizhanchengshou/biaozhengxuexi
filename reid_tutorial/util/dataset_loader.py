from __future__ import print_function, absolute_import
import os
from PIL import Image
from IPython import embed
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset

from reid_tutorial.util import data_manager


# 读取图片
def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

if __name__ == '__main__':
    dataset = data_manager.init_img_dataset(root='D:/xiangmu/reid/data', name='market1501')
    train_loader = ImageDataset(dataset.train)
    embed()
