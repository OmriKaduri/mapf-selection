import os
from pathlib import Path

import hyperopt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
from metrics import runtime_adjusted_coverage_score, normalized_coverage_score, normalized_accuracy_score, cumsum_score
from models.mapf_model import MapfModel
from preprocess import Preprocess
from joblib import dump, load


class CoverageClassifier(MapfModel):
    def __init__(self, *args):
        super(CoverageClassifier, self).__init__(*args)
        self.classifiers = []
        self.balanced = False
        self.xg_test_res = []
        self.xg_train_res = []
        self.modelname = 'XGBoost Coverage'
        if self.maptype != '':
            self.modelname += '-' + self.maptype

    def balance_dataset(self):
        self.balanced = True
        self.modelname += ' - balanced'

    def sample_weight(self, X_train):
        return [1] * len(X_train)

    def train_cv(self, data, exp_type, n_splits=2, hyperopt_evals=5, load=False,
                 model_suffix='-coverage-model.xgb',
                 models_dir='models/coverage'):

        self.classifiers = []
        if self.balanced:
            data = Preprocess.balance_dataset_by_label(data)

        if len(set(data['InstanceId'])) > 1:
            groups = data['InstanceId']  # len of scenarios
        else:
            groups = data.index
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
        only_alg_success_cols = [Preprocess.runtime_to_success(col) for col in self.only_alg_runtime_cols]
        for success_col in only_alg_success_cols:
            classifier = xgb.XGBClassifier(objective='binary:logistic')
            curr_model_path = str(model_path / (success_col.split(' ')[0] + model_suffix))
            if load and os.path.exists(curr_model_path):
                classifier.load_model(curr_model_path)
                print("loaded coverage model from", curr_model_path)
                self.classifiers.append(classifier)
                continue

            index += 1
            best_coverage = 0
            best_params = {}
            for index, (tr_ind, test_ind) in enumerate(gkf.split(data[self.features_cols], data[success_col], groups)):
                print("Starting {i} inner fold out of {n} in coverage training".format(i=index, n=n_splits))
                X_train = data.iloc[tr_ind].copy()
                y_train = data[success_col].iloc[tr_ind].copy()

                X_test = data.iloc[test_ind].copy()
                y_test = data[success_col].iloc[test_ind].copy()
                curr_best_params, trials = self.find_best_params(X_train, y_train, X_test, y_test,
                                                                 xgb.XGBClassifier,
                                                                 {'objective': 'binary:logistic',
                                                                  'scale_pos_weight': 1},
                                                                 param_dist,
                                                                 # {'sample_weight': self.sample_weight(X_train)},
                                                                 max_evals=hyperopt_evals, )

                fnvals = [(t['result']) for t in trials.trials]
                params = min(fnvals, key=lambda x: -x['loss'])
                if -params['loss'] > best_coverage:
                    best_coverage = -params['loss']
                    best_params = curr_best_params

            print("Best params", best_params)

            classifier = xgb.XGBClassifier(**best_params)
            classifier = classifier.fit(data[self.features_cols], data[success_col], sample_weight=train_samples_weight)

            self.classifiers.append(classifier)
            classifier.save_model(curr_model_path)

    def predict(self, X_test, y_test, online_feature_extraction_time=None):
        self.xg_test_res = []
        for classifier in self.classifiers:
            self.xg_test_res.append(np.array(classifier.predict_proba(X_test[self.features_cols]))[:, 1])

        self.xg_test_res = np.array(self.xg_test_res)

        test_preds = [self.conversions[index] for index in self.xg_test_res.argmax(axis=0)]

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
