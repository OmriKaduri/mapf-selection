import os
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, LeakyReLU, AvgPool2d, Flatten
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from torch.nn.functional import pad
from metrics import runtime_adjusted_coverage_score, coverage_score, cumsum_score
from preprocess import Preprocess
from pathlib import Path

import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from models.mapf_model import MapfModel
import matplotlib.pyplot as plt

from models.spp_layer import spatial_pyramid_pool


def npy_loader(path):
    sample = dict(np.load(path))['arr_0']
    sample = torch.from_numpy(sample)
    return sample


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def default_collate(batch):
    """
    Override `default_collate` https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader

    Reference:
    def default_collate(batch) at https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    https://github.com/pytorch/pytorch/issues/1512
    """
    data = [item[0].numpy() for item in batch]
    label = [item[1].numpy() for item in batch]  # image labels.

    return data, label


def get_padding(image):
    max_w = 224
    max_h = 224

    imsize = image.size()
    h_padding = (max_w - imsize[1]) / 2
    v_padding = (max_h - imsize[2]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5

    padding = [int(l_pad), int(t_pad), int(r_pad), int(b_pad)]

    return padding


def pad_image(image):
    padded_im = pad(image, get_padding(image))  # torchvision.transforms.functional.pad
    return padded_im


class MAPFDataset(Dataset):
    def __init__(self, df, success_cols, transforms=transforms.Compose([]), datafolder='nd_mapf'):
        self.df = df
        self.success_cols = success_cols
        self.datafolder = datafolder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        problem_name = row.GridName + '_' + str(row.InstanceId) + '_' + str(row.NumOfAgents) + '.npz'
        x = npy_loader(Path(self.datafolder) / problem_name)
        x = x.float()
        x = pad_image(x)
        x = transforms.Normalize(0, 1)(x)
        y = torch.tensor(row[self.success_cols])
        return x, y


class MapfClassifier(pl.LightningModule):
    def __init__(self, val_df, conversions, num_target_classes=4):
        super().__init__()
        self.num_target_classes = num_target_classes
        self.val_df = val_df
        self.conversions = conversions
        self.to_display = None
        features = 32
        self.output_num = [3, 2, 1]

        # self.feature_extractor = models.resnet18(
        #     pretrained=False)

        self.feature_extractor = Sequential(
            # Defining a 2D convolution layer
            # 2x224x224
            Conv2d(2, features, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(features),
            LeakyReLU(0.2, inplace=True),
            Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(features),
            LeakyReLU(0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # 32x112x112
            Conv2d(features, features * 2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(features * 2),
            LeakyReLU(0.2, inplace=True),
            Conv2d(features * 2, features * 2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(features * 2),
            LeakyReLU(0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # 64x56x56
            Conv2d(features * 2, features * 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(features * 4),
            LeakyReLU(0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(features * 4, features * 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(features * 4),
            LeakyReLU(0.2, inplace=True),
            Conv2d(features * 4, features * 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(features * 4),
            LeakyReLU(0.2, inplace=True),
            # 128x28x28
            # Conv2d(features * 4, features * 4, kernel_size=3, stride=1, padding=1),
            # BatchNorm2d(features * 4),
            # LeakyReLU(0.2, inplace=True),
            # MaxPool2d(kernel_size=2, stride=2),
            # 256x14x14
        )

        # self.feature_extractor.conv1 = torch.nn.Conv2d(2, 64, (3, 3), (1, 1), (1, 1))
        # self.feature_extractor.eval()

        self.classifier = Sequential(
            # nn.Linear(features * 2, 64),
            # LeakyReLU(0.2, inplace=True),
            nn.Linear(1792, 128),
            nn.Linear(128, num_target_classes)
        )

        self.test_preds = []

    def forward(self, x):
        features = self.feature_extractor(x)
        features = spatial_pyramid_pool(features, features.size(0), [int(features.size(2)), int(features.size(3))],
                                        self.output_num)
        # features = features.squeeze()
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch

        if self.to_display is None:
            self.to_display = x[0]
        y_hat = self.forward(x)
        y = y.type_as(y_hat)
        # loss = F.cross_entropy(y_hat, y)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        a, y_hat = torch.max(y_hat, dim=1)
        # acc = accuracy_score(y.clone().cpu().detach().numpy(), y_hat.cpu())
        tensorboard_logs = {'train_loss': loss,
                            # 'train_acc': acc
                            }
        return {'loss': loss, 'log': tensorboard_logs}

    def on_batch_end(self):
        if self.to_display is not None:
            self.logger.experiment.add_image("input-map", torch.Tensor.cpu(self.to_display[0]), self.current_epoch,
                                             dataformats="HW")
            self.logger.experiment.add_image("input-collisions", torch.Tensor.cpu(self.to_display[1]),
                                             self.current_epoch,
                                             dataformats="HW")
        self.to_display = None

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch

        y_hat = self.forward(x)
        y = y.type_as(y_hat)
        # loss = F.cross_entropy(y_hat, y)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        a, y_hat = torch.max(y_hat, dim=1)
        # acc = accuracy_score(y.clone().cpu().detach().numpy(), y_hat.cpu())
        return {'val_loss': loss,
                # 'val_acc': torch.tensor(acc),
                'preds': y_hat}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        preds = torch.cat([x['preds'] for x in outputs]).cpu().detach().numpy()
        preds = [self.conversions[x] for x in preds]
        if len(preds) != len(self.val_df):
            cov = 0.0
        else:
            cov = coverage_score(self.val_df, preds)
            print("Coverage:", cov)
        tensorboard_logs = {'val_loss': avg_loss,
                            # 'val_acc': avg_acc,
                            'val_cov': cov}
        return {'val_loss': avg_loss, 'val_voc': cov, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        y = y.type_as(y_hat)
        # loss = F.cross_entropy(y_hat, y)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        a, y_hat = torch.max(y_hat, dim=1)
        if len(self.test_preds) == 0:
            self.test_preds = y_hat
        else:
            self.test_preds = torch.cat([self.test_preds, y_hat])
        # acc = accuracy_score(y.clone().cpu().detach().numpy(), y_hat.cpu())
        return {'test_loss': loss,
                # 'test_acc': torch.tensor(acc)
                }

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss,
                            # 'test_acc': avg_acc
                            }
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0.1)
        return optimizer


class TorchClassifier(MapfModel):
    def __init__(self, *args):
        super(TorchClassifier, self).__init__(*args)
        self.modelname = 'PyTorch Classification'
        self.model = None
        self.trainer = pl.Trainer(max_epochs=5, gpus=1)
        self.conversions = dict(zip(np.arange(len(self.only_alg_runtime_cols)), iter(self.only_alg_runtime_cols)))
        if self.maptype != '':
            self.modelname += '-' + self.maptype

    def train_cv(self, data, labels, n_splits=2, hyperopt_evals=5, load=False, model_suffix='clf-model.xgb',
                 models_dir='models'):
        groups = data['InstanceId']  # len of scenarios
        gkf = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)

        for index, (tr_ind, test_ind) in enumerate(gkf.split(data[self.features_cols], labels, groups)):
            print("Starting {i} inner fold out of {n} in pytorch classification training".format(i=index, n=n_splits))
            self.model = MapfClassifier(val_df=data.iloc[test_ind].copy(),
                                        conversions=self.conversions,
                                        num_target_classes=len(self.conversions))
            print(self.model)
            train = MAPFDataset(data.iloc[tr_ind].copy(), success_cols=self.success_cols)
            val = MAPFDataset(data.iloc[test_ind].copy(), success_cols=self.success_cols)
            self.trainer.fit(self.model,
                             DataLoader(train, batch_size=128, num_workers=1, shuffle=True),
                             DataLoader(val, batch_size=128, num_workers=1, ))

    def predict(self, X_test, y_test, online_feature_extraction_time=None):
        test_dataset = MAPFDataset(X_test, success_cols=self.success_cols)
        self.trainer.test(self.model,
                          test_dataloaders=DataLoader(test_dataset, batch_size=128, num_workers=1))
        y_test = X_test['Y']
        test_preds = self.model.test_preds.cpu().detach().numpy()
        test_preds = [self.conversions[x] for x in test_preds]
        # model_acc = accuracy_score(y_test, test_preds)
        model_acc = 1.0
        if online_feature_extraction_time:
            model_coverage = runtime_adjusted_coverage_score(X_test, test_preds,
                                                             (self.max_runtime - X_test[
                                                                 online_feature_extraction_time]))
        else:
            model_coverage = coverage_score(X_test, test_preds, self.max_runtime)
        model_cumsum = cumsum_score(X_test, test_preds, online_feature_extraction_time)

        print(self.modelname, "Accuracy:", model_acc)
        print(self.modelname, "Coverage:", model_coverage)
        print(self.modelname, "Cumsum:", model_cumsum)

        self.results = self.results.append({'Model': self.modelname,
                                            'Accuracy': model_acc,
                                            'Coverage': model_coverage,
                                            'Cumsum': model_cumsum},
                                           ignore_index=True)

        return test_preds
