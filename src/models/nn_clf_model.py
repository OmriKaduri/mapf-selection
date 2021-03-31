import os
import torch
import yaml
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, LeakyReLU, AvgPool2d
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from torch.nn.functional import pad
from metrics import runtime_adjusted_coverage_score, coverage_score, cumsum_score
from preprocess import Preprocess
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from models.mapf_model import MapfModel
import matplotlib.pyplot as plt

from models.spp_layer import spatial_pyramid_pool


def create_8by8_from_row(row):
    mapf_representation = np.zeros((8, 8, 4))
    for i in range(0, 8):
        for j in range(0, 8):
            mapf_representation[i][j][0] = row[f'{i}_{j}_cell_agent_start'] * 10
            mapf_representation[i][j][1] = row[f'{i}_{j}_cell_agent_goal'] * 10
            mapf_representation[i][j][2] = row[f'{i}_{j}_cell_obstacles'] * 10
            mapf_representation[i][j][3] = row[f'{i}_{j}_cell_open'] * 10

    return torch.from_numpy(mapf_representation)


class MAPFDataset(Dataset):
    def __init__(self, df, features_cols):
        self.df = df
        self.features_cols = features_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = create_8by8_from_row(row)
        x = x.float()
        x = x.permute(2, 0, 1)
        # x = transforms.Normalize(mean=[0, 0, 0, 0],
        #                          std=[1, 1, 1, 1])(x)
        y = torch.tensor(row['Y_code'])
        manual_features = torch.from_numpy(row[self.features_cols].values.astype('float'))
        # min_v = torch.min(manual_features)
        # range_v = torch.max(manual_features) - min_v
        # normalised = (manual_features - min_v) / range_v
        # manual_features = transforms.Normalize(mean=[0],
        #                                        std=[1])(manual_features)
        return (x, manual_features), y


def set_parameter_requires_grad(model, feature_extracting, n_layers=20):
    if feature_extracting:
        c = 0
        for child in model.children():
            if c < n_layers:
                for param in child.parameters():
                    param.requires_grad = False
                c += 1


class NNMapfClassifier(pl.LightningModule):
    def __init__(self, val_df, conversions, features_cols, num_target_classes=4, max_runtime=300000):
        super().__init__()
        self.num_target_classes = num_target_classes
        self.val_df = val_df
        self.features_cols = features_cols
        self.conversions = conversions
        self.to_display = None
        self.max_runtime = max_runtime
        self.cnn = Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=1),
            # nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
            # nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1),
            # nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # Now 2x2x16
        )
        self.manual_features_route = Sequential(
            nn.Linear(len(self.features_cols), 64),
            nn.Sigmoid()
        )
        self.embed_with_manual_features = Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
        self.cls_head = nn.Linear(64, self.num_target_classes)
        self.test_preds = []

    def forward(self, x):
        img, manual_features = x
        features = self.cnn(img)
        features = torch.flatten(features, start_dim=1)  # Bx64
        # manual_features = self.manual_features_route(manual_features.float())
        # concat_features = torch.cat((features, manual_features.float()), 1)  # Bx64+len(manual_features)
        # concat_features = self.embed_with_manual_features(concat_features)
        output = self.cls_head(features)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        if self.to_display is None:
            self.to_display = x[0]
        y_hat = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        a, y_hat = torch.max(y_hat, dim=1)
        acc = accuracy_score(y.clone().cpu().detach().numpy(), y_hat.cpu())
        tensorboard_logs = {'train_loss': loss,
                            'train_acc': acc
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
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        a, y_hat = torch.max(y_hat, dim=1)
        acc = accuracy_score(y.clone().cpu().detach().numpy(), y_hat.cpu())
        return {'val_loss': loss,
                'val_acc': torch.tensor(acc),
                'preds': y_hat}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        preds = torch.cat([x['preds'] for x in outputs]).cpu().detach().numpy()
        preds = [self.conversions[x] for x in preds]
        if len(preds) != len(self.val_df):
            cov = 0.0
        else:
            cov = coverage_score(self.val_df, preds, self.max_runtime)
            print("Coverage:", cov)
        tensorboard_logs = {'val_loss': avg_loss,
                            'val_acc': avg_acc,
                            'val_cov': cov}
        return {'val_loss': avg_loss, 'val_voc': cov, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(0)
            print(y.shape, y_hat.shape)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)

        a, y_hat = torch.max(y_hat, dim=1)
        if len(self.test_preds) == 0:
            self.test_preds = y_hat
        else:
            self.test_preds = torch.cat([self.test_preds, y_hat])
        acc = accuracy_score(y.clone().cpu().detach().numpy(), y_hat.cpu())
        return {'test_loss': loss,
                'test_acc': torch.tensor(acc)
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
        for name, param in self.cnn.named_parameters():
            params_to_update.append(param)

        for name, param in self.embed_with_manual_features.named_parameters():
            params_to_update.append(param)

        for name, param in self.manual_features_route.named_parameters():
            params_to_update.append(param)

        for name, param in self.cls_head.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.0001, weight_decay=0.05)
        return optimizer


class NNClfModel(MapfModel):
    def __init__(self, *args):
        super(NNClfModel, self).__init__(*args)
        self.modelname = 'NN Classification'
        self.model = None
        if self.maptype != '':
            self.modelname += '-' + self.maptype
        self.conversions = dict(zip(np.arange(len(self.only_alg_runtime_cols)), iter(self.only_alg_runtime_cols)))
        self.features_cols = [f for f in self.features_cols if '_cell_' not in f]

    def train_cv(self, data, labels, exp_type, load=False, model_suffix='clf-model.pt',
                 models_dir='models/nn-classification/', n_splits=1):
        groups = data['InstanceId']  # len of scenarios
        gkf = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=41)
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

            self.trainer = pl.Trainer(max_epochs=30, gpus=0, callbacks=[early_stop_callback])
            print("Starting {i} inner fold out of {n} in nn classification training".format(i=index, n=n_splits))
            self.model = NNMapfClassifier(val_df=data.iloc[test_ind].copy(),
                                          conversions=self.conversions,
                                          features_cols=self.features_cols,
                                          num_target_classes=len(self.conversions),
                                          max_runtime=self.max_runtime)
            curr_model_path = str(model_path / model_suffix)
            if load and os.path.exists(curr_model_path):
                self.model.load_state_dict(torch.load(curr_model_path))
                self.model.eval()
                print("loaded nn-classification model from", curr_model_path)
                return
            train = MAPFDataset(data.iloc[tr_ind].copy(), features_cols=self.features_cols)
            val = MAPFDataset(data.iloc[test_ind].copy(), features_cols=self.features_cols)
            self.trainer.fit(self.model,
                             DataLoader(train, batch_size=128, num_workers=0, shuffle=True),
                             DataLoader(val, batch_size=128, num_workers=0, ))
            torch.save(self.model.state_dict(), curr_model_path)

    def predict(self, X_test, y_test, online_feature_extraction_time=None):
        test_dataset = MAPFDataset(X_test.copy(), features_cols=self.features_cols)
        self.trainer.test(self.model,
                          test_dataloaders=DataLoader(test_dataset, batch_size=128, num_workers=0))
        y_test = X_test['Y']
        test_preds = self.model.test_preds.cpu().detach().numpy()

        test_preds = [self.conversions[x] for x in test_preds]
        model_acc = accuracy_score(y_test, test_preds)

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
