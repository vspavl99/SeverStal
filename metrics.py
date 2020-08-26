import torch


def dice_single_channel(targets, preds, eps=1e-9):
    batch_size = preds.shape[0]
    preds = preds.view((batch_size, -1)).float()
    targets = targets.view((batch_size, -1)).float()
    dice = (2 * (preds * targets).sum(1) + eps) / (preds.sum(1) + targets.sum(1) + eps)
    return dice


def mean_dice_score(targets, outputs, threshold=0.5):
    batch_size = outputs.shape[0]
    n_channels = outputs.shape[1]
    preds = (outputs.sigmoid() > threshold).float()

    mean_dice = 0
    for i in range(n_channels):
        dice = dice_single_channel(targets[:, i, :, :], preds[:, i, :, :])
        mean_dice += dice.sum(0) / (n_channels * batch_size)
    return mean_dice.item()


def pixel_accuracy_score(targets, outputs, threshold=0.5):
    preds = (outputs.sigmoid() > threshold).float()
    correct = torch.sum((targets == preds)).item()
    total = outputs.numel()
    return correct / total


def epoch_metrics(targets, outputs, threshold=0.5):
    return {'dice': mean_dice_score(targets, outputs, threshold),
            'pixel_acc': pixel_accuracy_score(targets, outputs, threshold)}


def predict(output, threshold=0.5):
    prediction = (output.sigmoid() > threshold).float()
    return prediction


