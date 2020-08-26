import pandas as pd
import torch
import os
import cv2
from dataset import get_transforms
from metrics import predict
from utils import mask_to_rle
import segmentation_models_pytorch as smp

def test_generator(transforms, path_to_images='test_images'):
    images_name = os.listdir(path_to_images)

    for name in images_name:
        image = cv2.imread(os.path.join(path_to_images, name))
        image = transforms(image=image)['image']
        yield image, name


def make_submission(
        model,
        path_to_images='test_images',
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    model.to(device)
    transforms = get_transforms(phase='val')
    test_images = test_generator(transforms, path_to_images)
    result = pd.DataFrame(columns=['ImageId', 'EncodedPixels', 'ClassId'])
    for image, name in test_images:
        output = predict(model(image.unsqueeze(0).to(device)).cpu().detach())
        print((output != 0).sum())
        for defect in range(4):
            rle = mask_to_rle(output[:, defect, :, :])
            result = result.append({'ImageId': name, 'EncodedPixels': rle, 'ClassId': defect + 1}, ignore_index=True)

    return result


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet('resnet18', classes=4, activation=None)
    model.load_state_dict(torch.load('models/model_epoch_19_score_0.8166826734999697.pth', map_location=device)['state_dict'])
    result = make_submission(model, path_to_images='train_images')
    result.to_csv("submission.csv", index=False)
    print(result)