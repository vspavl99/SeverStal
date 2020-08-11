import torch


def dice_coef(predicted, truth, eps=1e-9):
    intersection = (predicted * truth).sum()
    dice = 2 * intersection / (predicted.sum() + truth.sum() + eps)
    return dice


def predict(output, threshold=0.5):
    output = torch.sigmoid(output)
    output[output >= threshold] = 1
    output[output < threshold] = 0
    return output
