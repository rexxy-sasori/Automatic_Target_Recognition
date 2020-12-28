"""
Author: Rex Geng

class definition for dataset
"""

import glob
import os

import torchvision
from torch.utils.data.dataset import Dataset

from vision import img_processing


class ATRDataset(Dataset):
    def __init__(self,
                 classes,
                 dataset_dir,
                 train=True,
                 transform=None,
                 distort=False,
                 crop_size=0):
        super(ATRDataset, self).__init__()

        self.transforms = transform
        self.classes = classes
        self.dataset_dir = dataset_dir
        self.labels = []
        self.imgs_dir = []
        self.augment = []
        self.distort = distort
        self.crop_size = crop_size

        for cls_idx, cls in enumerate(classes):
            if train:
                class_dir = os.path.join(dataset_dir, 'train/CONVERT_DATA_' + cls + '_17deg/')
            else:
                class_dir = os.path.join(dataset_dir, 'test/CONVERT_DATA_' + cls + '_15deg/')

            png_image_files = glob.glob(os.path.join(class_dir, '*.png'))
            jpg_image_files = glob.glob(os.path.join(class_dir, '*.JPG'))
            image_files = png_image_files + jpg_image_files

            for image_file in image_files:
                self.imgs_dir.append(image_file)
                self.labels.append(cls_idx)
                self.augment.append('o')

                if distort:
                    self.imgs_dir.append(image_file)
                    self.labels.append(cls_idx)
                    self.augment.append('ed_1')

                    self.imgs_dir.append(image_file)
                    self.labels.append(cls_idx)
                    self.augment.append('ed_2')

    def __getitem__(self, index):
        assert 0 <= index < len(self.labels)
        img = img_processing.img_read(self.imgs_dir[index], crop_size=self.crop_size)
        lbl = self.labels[index]
        pre_flag = self.augment[index]

        if pre_flag == 'ed_1':
            img = img_processing.elastic_transform(img, alpha=10, sigma=3)
        elif pre_flag == 'ed_2':
            img = img_processing.elastic_transform(img, alpha=20, sigma=3)

        img = img / 256.0

        if self.transforms:
            img = self.transforms(img)

        return img, lbl

    def __len__(self):
        return len(self.labels)


__DATASET__ = {
    'atr': ATRDataset,
    'cifar10': torchvision.datasets.CIFAR10
}
