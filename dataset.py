from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
import os
import numpy as np

from utils import rle_to_mask
from utils import show_defects
import albumentations as albu
import albumentations.pytorch as albu_pytorch


class SteelDataset(Dataset):
    def __init__(self, dataset, phase='train', dir='train_images', image_size=(256, 1600), n_classes=4):
        self.dataset = dataset
        self.phase = phase
        self.dir = dir
        self.transforms = get_transforms()
        self.image_size = image_size
        self.n_classes = n_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        name = self.dataset['ImageId'].iloc[index]
        image = cv2.imread(os.path.join(self.dir, name))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.zeros((self.n_classes, self.image_size[0], self.image_size[1]))
        defects = self.dataset[self.dataset['ImageId'] == name]['ClassId'].values
        for defect in defects:
            rle = self.dataset[(self.dataset['ImageId'] == name) &
                               (self.dataset['ClassId'] == defect)]['EncodedPixels'].values[0]
            encoded = rle_to_mask(rle)
            mask[defect - 1, :] = encoded

        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']

        return image, mask


def get_transforms(list_transforms=None, phase='train'):
    if not list_transforms:
        list_transforms = []

    if phase == 'train':
        list_transforms.extend(
            [
                albu.RandomBrightnessContrast(p=0.1, brightness_limit=0.1, contrast_limit=0.1),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                # albu.ElasticTransform(p=0.5),
                # albu.GridDistortion(p=0.5),
                # albu.OpticalDistortion(p=0.5),
            ]
        )
    list_transforms.extend(
        [
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albu_pytorch.ToTensorV2()
        ]
    )

    list_transforms = albu.Compose(list_transforms)
    return list_transforms


if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    dataset = SteelDataset(df)

    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    for i, (image, mask) in enumerate(dataloader):
        print(i)
        show_defects(image, mask)
