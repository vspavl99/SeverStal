import torch
from torch import nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import SteelDataset
import utils
import pandas as pd


model = smp.Unet('resnet18', encoder_weights='imagenet', classes=4, activation=None)
# input size of image (num_bathes, channels, width, higher)


df = pd.read_csv('train.csv')

dataset = SteelDataset(df)
dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
criterion = nn.BCEWithLogitsLoss()

for i, (image, mask) in enumerate(dataloader):
    print(i)
    pred = model.predict(image)
    for n in range(8):
        # show_defects(image[n], mask[n])
        # show_defects(image[n], pred[n])
        utils.show_mask_image(image[n], mask[n])
        utils.show_mask_image(image[n], pred[n])
        # print(criterion(pred[n], mask[n]))