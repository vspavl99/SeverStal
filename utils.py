import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd


def mask_to_rle(mask):
    """
    mask:  numpy array,  1 - mask, 0 - background
    return: run length as string formatted
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[:-1] != pixels[1:])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_to_mask(rle, shape=(1600, 256)):
    """
    :param rle: run-length as string formated (start length)
    :param shape: (width,height) of array to return
    :return: numpy array, 1 - mask, 0 - background
    """
    runs = np.array([int(x) for x in rle.split()])
    runs[1::2] += runs[::2]
    runs -= 1
    starts, ends = runs[::2], runs[1::2]
    mask = np.zeros(shape[0] * shape[1])
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape(shape).T


def show_defects(image, mask, pallet=((249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12))):

    for i in range(4):
        image[0, mask[i] == 1] = 255
    plt.imshow(image.permute(1, 2, 0))
    plt.show()


def show_mask_image(image, mask, pallet=((249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12))):
    fig, ax = plt.subplots(figsize=(15, 15))
    image = image.permute(1, 2, 0).numpy()
    mask = mask.permute(1, 2, 0)

    for ch in range(4):
        image[mask[:, :, ch] == 1] = pallet[ch]
    plt.imshow(image)

    plt.show()


def make_mask(name, df):
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)
    rows = df.loc[name]
    for defect in range(1, 5):
        rle = rows[defect]
        if not pd.isna(rle):
            encoded = rle_to_mask(rle)
            mask[:, :, defect - 1] = encoded
    return mask

