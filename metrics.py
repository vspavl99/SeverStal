import torch


def dice_coef(predicted, truth, eps=1e-9):
    intersection = (predicted * truth).sum()
    dice = 2 * intersection / (predicted.sum() + truth.sum() + eps)
    return dice

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

def predict(output, threshold=0.5):
    output = torch.sigmoid(output)
    output[output >= threshold] = 1
    output[output < threshold] = 0
    return output


if __name__ == '__main__':
    a = torch.tensor([
        [[[0, 0, 0],
          [0, 0, 0]],

         [[1, 1, 1],
          [0, 1, 1]]],

        [[[0, 0, 0],
          [0, 0, 0]],

         [[1, 1, 1],
          [0, 1, 1]]]
    ])
    b = torch.tensor([
        [[[1, 0, 0],
          [0, 1, 1]],

         [[1, 1, 1],
          [0, 1, 1]]],

        [[[1, 0, 0],
          [0, 1, 1]],

         [[1, 1, 1],
          [0, 1, 1]]]
    ])

    for i in range(100):
        a, b = torch.rand((4, 3, 256, 1600)), torch.rand((4, 3, 256, 1600))
        print(dice_coef(a, b).item(), mean_dice_score(a, b))
        # print(dice_coef(a, b).item() == dice_single_channel(a, b).mean().item())
        # assert dice_coef(a, b) == dice_single_channel(a, b).mean()



