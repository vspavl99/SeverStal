import torch
from torch import nn
import segmentation_models_pytorch as smp
from dataset import data_provider
from metrics import dice_coef
import pandas as pd
import time
from tqdm import tqdm_notebook as tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, data_frame, num_epochs, batch_size=8):
        self.model = model
        self.device = device
        self.model = model.to(device)
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size
        self.df = data_frame
        self.dataloaders = data_provider(self.df, batch_size=self.batch_size)
        self.losses = {phase: [] for phase in ['train', 'val']}
        self.metrics = {'dice': dice_coef}
        self.metrics_values = {phase: {name : [] for name in self.metrics.keys()}
                               for phase in ['train', 'val']}

    def step(self, epoch, phase):
        epoch_loss = 0.0
        epoch_metric = 0.0
        if phase == 'train':
            model.train()
        else:
            model.eval()

        dataloader = self.dataloaders[phase]

        for i, (images, targets) in enumerate(dataloader):
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                if phase == 'train':  # Train step
                    loss.backward()
                    self.optimizer.step()

                epoch_metric += self.metrics['dice'](outputs, targets) / len(dataloader)
                epoch_loss += loss.item() / len(dataloader)

        self.losses[phase].append(epoch_loss)
        self.metrics_values[phase]['dice'].append(epoch_metric)
        del images, targets, outputs, loss
        torch.cuda.empty_cache()

    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            self.step(epoch, 'train')
            state = {'epoch': epoch,
                     'best_score': self.best_score,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
            with torch.no_grad():
                self.step(epoch, 'val')


if __name__ == '__main__':
    epochs = 10

    model = smp.Unet('resnet18', encoder_weights='imagenet', classes=4, activation=None)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4, lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, mode="min", patience=3, verbose=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('train.csv')
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['NumDefects'] = df.count(axis=1)

    model_train = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        batch_size=1,
        num_epochs=2,
        data_frame=df
    )

    model_train.step(1, 'train')
    start = time.strftime('%H:%M:%S')
    print(start)
    model(torch.rand((4, 3, 256, 1600)))
    print(time.strftime('%H:%M:%S'))