import os
from pathlib import Path

import hyperopt
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
from metrics import runtime_adjusted_coverage_score, normalized_coverage_score, normalized_accuracy_score, cumsum_score
from models.mapf_model import MapfModel
from preprocess import Preprocess
import itertools
import pandas as pd


def is_first_model_win(row, first_model, second_model):
    if row[first_model] < row[second_model]:
        return True
    return False


class CostSensitiveClassifier(MapfModel):
    def __init__(self, *args):
        super(CostSensitiveClassifier, self).__init__(*args)
        self.classifiers = {}
        self.balanced = False
        self.xg_test_res = []
        self.xg_train_res = []
        self.modelname = 'Cost-Sensitive Coverage'
        if self.maptype != '':
            self.modelname += '-' + self.maptype

    def balance_dataset(self):
        self.balanced = True
        self.modelname += ' - balanced'

    def sample_weight(self, X_train):
        return [1] * len(X_train)

    def train_cv(self, data, exp_type, n_splits=2, hyperopt_evals=5, load=False,
                 model_suffix='-cost-model.xgb',
                 models_dir='models/cost-sensitive'):
        self.classifiers = {}
        if self.balanced:
            data = Preprocess.balance_dataset_by_label(data)

        gkf = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
        param_dist = {'n_estimators': hyperopt.hp.choice('n_estimators', np.arange(75, 500, 25)),
                      'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 0.07),
                      'subsample': hyperopt.hp.uniform('subsample', 0.3, 0.7),
                      'max_depth': hyperopt.hp.choice('max_depth', np.arange(3, 10)),
                      'min_child_weight': hyperopt.hp.choice('min_child_weight', np.arange(1, 4)),
                      "gamma": hyperopt.hp.choice('gamma', np.arange(0.1, 2, 0.2)),
                      "reg_alpha": hyperopt.hp.choice('reg_alpha', np.arange(0, 1.5, 0.5))}

        index = 0
        model_path = Path(models_dir) / exp_type
        model_path.mkdir(parents=True, exist_ok=True)

        train_samples_weight = self.sample_weight(data)
        for runtime_col, other_runtime_col in itertools.combinations(self.only_alg_runtime_cols, 2):
            index += 1
            best_coverage = 0
            best_params = {}
            classifier = xgb.XGBClassifier(objective='binary:logistic')
            label_name = runtime_col + '-vs-' + other_runtime_col
            curr_model_path = str(model_path / (label_name + model_suffix))
            if load and os.path.exists(curr_model_path):
                classifier.load_model(curr_model_path)
                print("loaded cost-sensitive model from", curr_model_path)
            else:
                data[label_name] = data.apply(
                    lambda x: is_first_model_win(x, runtime_col, other_runtime_col), axis=1)
                data[label_name + '_s_weight'] = data.apply(lambda x: np.abs(x[runtime_col] - x[other_runtime_col]),
                                                            axis=1)
                # 1 - first alg won, 0 - second alg won
                curr_data = data.copy()

                curr_data = self.remove_unsolved_problems_by_both_algorithms(curr_data, other_runtime_col, runtime_col)
                if len(set(curr_data['InstanceId'])) > 1:
                    groups = curr_data['InstanceId']  # len of scenarios
                else:
                    groups = curr_data.index

                for index, (tr_ind, test_ind) in enumerate(
                        gkf.split(curr_data[self.features_cols], curr_data[label_name], groups)):
                    print("Starting {i} inner fold out of {n} in cost-sensitive training".format(i=index, n=n_splits))

                    X_train = curr_data.iloc[tr_ind].copy()
                    y_train = curr_data[label_name].iloc[tr_ind].copy()

                    X_test = curr_data.iloc[test_ind].copy()
                    y_test = curr_data[label_name].iloc[test_ind].copy()
                    sample_weights = X_train[label_name + '_s_weight'].values
                    # sample_weights = np.random.randint(0, 1, len(X_train))
                    sample_weights = np.zeros(len(X_train))
                    curr_best_params, trials = self.find_best_params(X_train, y_train, X_test, y_test,
                                                                     xgb.XGBClassifier,
                                                                     {'objective': 'binary:logistic',
                                                                      # 'scale_pos_weight': 1
                                                                      },
                                                                     param_dist,
                                                                     {'sample_weight': sample_weights},
                                                                     max_evals=hyperopt_evals, )

                    fnvals = [(t['result']) for t in trials.trials]
                    params = min(fnvals, key=lambda x: -x['loss'])
                    if -params['loss'] > best_coverage:
                        best_coverage = -params['loss']
                        best_params = curr_best_params

                print("Best params", best_params)
                classifier = xgb.XGBClassifier(**best_params)
                classifier = classifier.fit(data[self.features_cols], data[label_name],
                                            sample_weight=train_samples_weight)
                classifier.save_model(curr_model_path)

            if runtime_col in self.classifiers:
                self.classifiers[runtime_col][other_runtime_col] = classifier
            else:
                self.classifiers[runtime_col] = {}
                self.classifiers[runtime_col][other_runtime_col] = classifier

    def remove_unsolved_problems_by_both_algorithms(self, curr_data, other_runtime_col, runtime_col):
        return curr_data[(curr_data[runtime_col] < 300000) | (curr_data[other_runtime_col] < 300000)]

    def predict(self, X_test, y_test, online_feature_extraction_time=None):
        self.wins = pd.DataFrame(0, index=np.arange(len(X_test)), columns=self.only_alg_runtime_cols)
        for runtime_col, other_runtime_col in itertools.combinations(self.only_alg_runtime_cols, 2):
            preds = np.array(
                self.classifiers[runtime_col][other_runtime_col].predict(X_test[self.features_cols])).astype(int)
            self.wins[runtime_col] += preds
            self.wins[other_runtime_col] += 1 - preds

        test_preds = list(self.wins.idxmax(1).values)

        model_acc = normalized_accuracy_score(X_test, test_preds)
        if online_feature_extraction_time:
            model_coverage = runtime_adjusted_coverage_score(X_test, test_preds, (self.max_runtime - X_test[
                online_feature_extraction_time]))
        else:
            model_coverage = normalized_coverage_score(X_test, test_preds, self.max_runtime)

        model_cumsum = cumsum_score(X_test, test_preds, online_feature_extraction_time)
        print(self.modelname, "Normalized Accuracy:", model_acc)
        print(self.modelname, "Normalized Coverage:", model_coverage)
        print(self.modelname, "Cumsum:", model_cumsum)

        self.results = self.results.append({'Model': self.modelname,
                                            'Normalized Accuracy': model_acc,
                                            'Normalized Coverage': model_coverage,
                                            'Cumsum': model_cumsum},
                                           ignore_index=True)

        return test_preds
