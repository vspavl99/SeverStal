from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
import os
import albumentations as albu
import albumentations.pytorch as albu_pytorch
from utils import make_mask
from sklearn.model_selection import train_test_split


class SteelDataset(Dataset):
    def __init__(self, dataset, phase='train', data_dir='train_images', image_size=(256, 1600), n_classes=4):
        self.dataset = dataset
        self.phase = phase
        self.dir = data_dir
        self.transforms = get_transforms(phase=self.phase)
        self.image_size = image_size
        self.n_classes = n_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        name = self.dataset.iloc[index].name
        image = cv2.imread(os.path.join(self.dir, name))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = make_mask(name, self.dataset)

        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask'].permute(2, 0, 1)

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


def data_provider(df, batch_size=8, shuffle=True, stratify_by=None):

    if stratify_by:
        train_df, val_df = train_test_split(df, test_size=0.2,
                                            stratify_by=df[stratify_by],
                                            random_state=42,
                                            shuffle=shuffle)
    else:
        train_df, val_df = train_test_split(df, test_size=0.2,
                                            random_state=42,
                                            shuffle=shuffle)

    dataloader = {'train': DataLoader(SteelDataset(train_df, phase='train'), batch_size=batch_size),
                  'val': DataLoader(SteelDataset(val_df, phase='val'), batch_size=batch_size)}


    return dataloader


if __name__ =='__main__':
    df = pd.read_csv('train.csv')
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['NumDefects'] = df.count(axis=1)
    dataloader = data_provider(df)['train']
    for i in dataloader:
        print((i[1] == 1).sum())