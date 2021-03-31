import os
import torch
import yaml
from PIL import Image
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
from metrics import runtime_adjusted_coverage_score, normalized_coverage_score, normalized_accuracy_score, cumsum_score
from preprocess import Preprocess
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from models.mapf_model import MapfModel
import matplotlib.pyplot as plt
import itertools
from models.spp_layer import spatial_pyramid_pool


def npy_loader(path):
    sample = dict(np.load(path, allow_pickle=True))['arr_0']
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
    _, w, h = image.size()
    max_wh = np.max([w, h])

    imsize = image.size()
    h_padding = (max_wh - imsize[1]) / 2
    v_padding = (max_wh - imsize[2]) / 2
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
    def __init__(self, df, success_cols, transforms=transforms.Compose([]), datafolder='custom_mapf_images'):
        self.df = df
        self.success_cols = success_cols
        self.datafolder = datafolder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        problem_name = row.GridName + '-' + row.problem_type + '-' + str(row.InstanceId) + '-' + str(
            row.NumOfAgents) + '.npz'
        x = npy_loader(Path(self.datafolder) / problem_name)
        x = x.float()
        x = x.permute(2, 0, 1)
        x = pad_image(x)
        x = F.interpolate(x.unsqueeze(0), size=(224, 224)).squeeze(0)
        x = transforms.Normalize(mean=[0, 0, 0],
                                 std=[1, 1, 1])(x)
        y = torch.tensor(row['pairs']).float()
        return x, y


def set_parameter_requires_grad(model, feature_extracting, n_layers=20):
    if feature_extracting:
        c = 0
        for child in model.children():
            if c < n_layers:
                for param in child.parameters():
                    param.requires_grad = False
                c += 1


def determine_winner(n):
    def calc(pair_preds):
        indices = list(itertools.combinations(np.arange(n), 2))
        alg_wins = [0] * n
        for index, pair_result in zip(indices, pair_preds):
            alg_wins[index[pair_result]] += 1  # Index is a tuple of (s1,s2) and we want to choose
            # the algorithm who won between them
        return np.array(alg_wins).argmax()

    return calc


class CNNMapfCostSensitive(pl.LightningModule):
    def __init__(self, val_df, conversions, num_target_classes=4, max_runtime=300000):
        super().__init__()
        self.num_target_classes = num_target_classes
        self.num_outputs = int((num_target_classes * (num_target_classes - 1)) / 2)  # NumCombinations = N*(N-1)/2
        # First output is for first algorithm vs second, N-1 output is for first algorithm vs last,
        # N output is for second algorithm vs third, ... (N*(N-1)/2) output is for second-to-last vs last algorithm
        self.val_df = val_df
        self.conversions = conversions
        self.to_display = None
        self.max_runtime = max_runtime
        self.feature_extractor = models.vgg16(pretrained=True).features
        set_parameter_requires_grad(self.feature_extractor, True)
        # In forward - GAP layer is used
        self.classifier = Sequential(
            nn.Linear(512, 64),
            nn.Linear(64, self.num_outputs)
        )
        self.test_preds = []

    def forward(self, x):
        features = self.feature_extractor(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.squeeze()
        output = self.classifier(features)
        output = F.sigmoid(output)  # We use sigmoid since we want a classifier for each algorithm pairs
        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        if self.to_display is None:
            self.to_display = x[0]
        y_hat = self.forward(x)
        if len(y_hat.shape) == 1:
            y_hat = y_hat.unsqueeze(0)
        criterion = nn.BCELoss()
        loss = criterion(y_hat, y)
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
        criterion = nn.BCELoss()
        loss = criterion(y_hat, y)
        # acc = accuracy_score(y.clone().cpu().detach().numpy(), y_hat.cpu())
        p = (y_hat > 0.5).cpu()  # For each pair's sigmoid output, apply threshold in-order to know who won
        return {'val_loss': loss,
                # 'val_acc': torch.tensor(acc),
                'preds': p}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        preds = torch.cat([x['preds'] for x in outputs]).cpu().detach().numpy()
        f = determine_winner(n=self.num_target_classes)
        preds = list(map(f, preds))
        preds = [self.conversions[x] for x in preds]
        if len(preds) != len(self.val_df):
            cov = 0.0
        else:
            cov = normalized_coverage_score(self.val_df, preds, self.max_runtime)
            print("Coverage:", cov)
        tensorboard_logs = {'val_loss': avg_loss,
                            # 'val_acc': avg_acc,
                            'val_cov': cov}
        return {'val_loss': avg_loss, 'val_voc': cov, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(0)
            print(y.shape, y_hat.shape)

        criterion = nn.BCELoss()
        loss = criterion(y_hat, y)

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
        print("Params to learn:")
        params_to_update = []
        for name, param in self.classifier.named_parameters():
            params_to_update.append(param)

        for name, param in self.feature_extractor.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.0001, weight_decay=0.05)
        return optimizer


def pairs_wins(row, pairs):
    wins = []
    for pair in pairs:
        if row[pair[0]] < row[pair[1]]:  # If first algorithm's cost is lower then second's
            wins.append(False)
        else:
            wins.append(True)
    return wins


class CNNCostSensitiveModel(MapfModel):
    def __init__(self, *args):
        super(CNNCostSensitiveModel, self).__init__(*args)
        self.modelname = 'CNN Cost Sensitive'

        self.model = None
        if self.maptype != '':
            self.modelname += '-' + self.maptype
        self.conversions = dict(zip(np.arange(len(self.only_alg_runtime_cols)), iter(self.only_alg_runtime_cols)))
        self.alg_combinations = list(itertools.combinations(self.only_alg_runtime_cols, 2))
        # TODO: Major problem with CNN Cost-sensitive is that some of the pairs are irrelevant in cost terms
        # (since both algorithms failed and recieved the same cost)
        # At regular cost sensitive, each classifier is trained only on the relevant data for each pair

    def train_cv(self, data, labels, exp_type, load=False, model_suffix='cost-model.pt',
                 models_dir='models/cost-sensitive/', n_splits=1):
        if len(set(data['InstanceId'])) > 1:
            groups = data['InstanceId']  # len of scenarios
        else:
            groups = data.index
        gkf = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=41)
        data['pairs'] = data.apply(lambda x: pairs_wins(x, self.alg_combinations),
                                   axis=1)  # Boolean array indicating for each pair who won
        model_path = Path(models_dir) / exp_type
        model_path.mkdir(parents=True, exist_ok=True)
        for index, (tr_ind, test_ind) in enumerate(gkf.split(data[self.features_cols], labels, groups)):
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0.00,
                patience=2,
                verbose=False,
                mode='min'
            )

            self.trainer = pl.Trainer(max_epochs=10, gpus=1, callbacks=[early_stop_callback])
            print("Starting {i} inner fold out of {n} in cnn cost-sensitive training".format(i=index, n=n_splits))
            self.model = CNNMapfCostSensitive(val_df=data.iloc[test_ind].copy(),
                                              conversions=self.conversions,
                                              num_target_classes=len(self.conversions),
                                              max_runtime=self.max_runtime)
            curr_model_path = str(model_path / model_suffix)
            if load and os.path.exists(curr_model_path):
                self.model.load_state_dict(torch.load(curr_model_path))
                self.model.eval()
                print("loaded cnn cost-sensitive model from", curr_model_path)
                return

            train = MAPFDataset(data.iloc[tr_ind].copy(), success_cols=self.success_cols)
            val = MAPFDataset(data.iloc[test_ind].copy(), success_cols=self.success_cols)
            self.trainer.fit(self.model,
                             DataLoader(train, batch_size=128, num_workers=5, shuffle=True),
                             DataLoader(val, batch_size=128, num_workers=5, ))
            torch.save(self.model.state_dict(), curr_model_path)

    def predict(self, X_test, y_test, online_feature_extraction_time=None):
        X_test['pairs'] = X_test.apply(lambda x: pairs_wins(x, self.alg_combinations),
                                       axis=1)  # Boolean array indicating for each pair who won
        test_dataset = MAPFDataset(X_test, success_cols=self.success_cols)
        self.trainer.test(self.model,
                          test_dataloaders=DataLoader(test_dataset, batch_size=128, num_workers=5))
        y_test = X_test['Y']
        test_preds = (self.model.test_preds > 0.5).cpu().detach().numpy()
        f = determine_winner(n=len(self.conversions))
        test_preds = list(map(f, test_preds))

        test_preds = [self.conversions[x] for x in test_preds]
        model_acc = normalized_accuracy_score(X_test, test_preds)

        if online_feature_extraction_time:
            model_coverage = runtime_adjusted_coverage_score(X_test, test_preds,
                                                             (self.max_runtime - X_test[
                                                                 online_feature_extraction_time]))
        else:
            model_coverage = normalized_coverage_score(X_test, test_preds, self.max_runtime)
        model_cumsum = cumsum_score(X_test, test_preds, online_feature_extraction_time)

        print(self.modelname, "Accuracy:", model_acc)
        print(self.modelname, "Normalized Coverage:", model_coverage)
        print(self.modelname, "Normalized Cumsum:", model_cumsum)

        self.results = self.results.append({'Model': self.modelname,
                                            'Normalized Accuracy': model_acc,
                                            'Normalized Coverage': model_coverage,
                                            'Cumsum': model_cumsum},
                                           ignore_index=True)

        return test_preds
