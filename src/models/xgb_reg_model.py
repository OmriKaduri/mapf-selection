import os
from pathlib import Path

import hyperopt
# import shap

from models.mapf_model import MapfModel
import numpy as np
from sklearn.metrics import accuracy_score
from metrics import normalized_coverage_score, normalized_accuracy_score, cumsum_score, runtime_adjusted_coverage_score
from sklearn.compose import TransformedTargetRegressor
import xgboost as xgb
from preprocess import Preprocess
from sklearn.model_selection import GroupShuffleSplit
from mapf_eda import MapfEDA
import matplotlib.pyplot as plt
from HyperoptObjective import func, inverse_func


def load_transformed_regressor(curr_model_path, xg_reg_trans, data, label):
    xg_reg_trans.regressor.load_model(curr_model_path)
    xg_reg_trans.regressor_ = xg_reg_trans.regressor
    # Line above is due to a check inside TransformedTargetRegressor for attrs with '_'
    xg_reg_trans._training_dim = data[label].ndim
    xg_reg_trans._fit_transformer(np.array(data[label]).reshape(-1, 1))


class XGBRegModel(MapfModel):

    def __init__(self, *args):
        super(XGBRegModel, self).__init__(*args)
        self.xg_regs = []
        self.balanced = False
        self.xg_test_res = []
        self.xg_train_res = []
        self.modelname = 'XGBoost Regression'
        if self.maptype != '':
            self.modelname += '-' + self.maptype

    def balance_dataset(self):
        self.balanced = True
        self.modelname += ' - balanced'

    def train_cv(self, data, exp_type, n_splits=2, hyperopt_evals=5, load=False, model_suffix='-reg-model.xgb',
                 models_dir='models/regression'):
        self.xg_regs = []
        if self.balanced:
            data = Preprocess.balance_dataset_by_label(data)

        if len(set(data['InstanceId'])) > 1:
            groups = data['InstanceId']  # len of scenarios
        else:
            groups = data.index

        gkf = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
        model_path = Path(models_dir) / exp_type
        model_path.mkdir(parents=True, exist_ok=True)

        param_dist = {'n_estimators': hyperopt.hp.choice('n_estimators', np.arange(75, 500, 25)),
                      'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 0.07),
                      'subsample': hyperopt.hp.uniform('subsample', 0.3, 0.7),
                      'max_depth': hyperopt.hp.choice('max_depth', np.arange(3, 10)),
                      'min_child_weight': hyperopt.hp.choice('min_child_weight', np.arange(1, 4)),
                      "gamma": hyperopt.hp.choice('gamma', np.arange(0.1, 2, 0.2)),
                      "reg_alpha": hyperopt.hp.choice('reg_alpha', np.arange(0, 1.5, 0.5))
                      }

        index = 0

        train_samples_weight = self.sample_weight(data)
        for runtime_col in self.only_alg_runtime_cols:
            xg_reg = xgb.XGBRegressor(objective='reg:squarederror')
            xg_reg_trans = TransformedTargetRegressor(regressor=xg_reg, func=func,
                                                      inverse_func=inverse_func)
            curr_model_path = str(model_path / (runtime_col.split(' ')[0] + model_suffix))
            curr_model_path = str(curr_model_path)
            if load and os.path.exists(curr_model_path):
                load_transformed_regressor(curr_model_path, xg_reg_trans, data, label=runtime_col)
                self.xg_regs.append(xg_reg_trans)
                print("loaded regression model for", runtime_col, "from", curr_model_path)
                continue

            index += 1
            best_mse = 9999999
            best_params = {}
            for index, (tr_ind, test_ind) in enumerate(gkf.split(data[self.features_cols], data[runtime_col], groups)):
                print("Starting {i} inner fold out of {n} in regression training".format(i=index, n=n_splits))
                X_train = data.iloc[tr_ind].copy()
                y_train = data[runtime_col].iloc[tr_ind].copy()

                X_test = data.iloc[test_ind].copy()
                y_test = data[runtime_col].iloc[test_ind].copy()
                curr_best_params, trials = self.find_best_params(X_train, y_train, X_test, y_test,
                                                                 None,
                                                                 {},
                                                                 param_dist,
                                                                 {'sample_weight': self.sample_weight(X_train)},
                                                                 max_evals=hyperopt_evals,
                                                                 is_regressor=True)

                fnvals = [(t['result']) for t in trials.trials]
                params = min(fnvals, key=lambda x: x['loss'])
                if params['loss'] < best_mse:
                    best_mse = params['loss']
                    best_params = curr_best_params

            print("Best params", best_params)
            xg_reg = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
            xg_reg_trans = TransformedTargetRegressor(regressor=xg_reg, func=func,
                                                      inverse_func=inverse_func)
            reg = xg_reg_trans.fit(data[self.features_cols], data[runtime_col],
                                   sample_weight=train_samples_weight,
                                   )
            reg.regressor_.save_model(curr_model_path)
            self.xg_regs.append(reg)

    def add_regression_as_features_to_data(self, X_train, X_test):
        features_with_reg_cols = self.features_cols.copy()
        for key, conversion in self.conversions.items():
            X_test['P ' + conversion] = self.xg_test_res[key]
            X_train['P ' + conversion] = self.xg_train_res[key]
            features_with_reg_cols.append('P ' + conversion)

        return X_train, X_test, features_with_reg_cols

    def predict(self, X_test, y_test, online_feature_extraction_time=None):
        self.xg_test_res = []
        for xg_reg in self.xg_regs:
            self.xg_test_res.append(np.array(xg_reg.predict(X_test[self.features_cols])))
        self.xg_test_res = np.array(self.xg_test_res)

        test_preds = [self.conversions[index] for index in self.xg_test_res.argmin(axis=0)]
        # self.X_test['P'] = test_preds
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

    def plot_feature_importance(self, df, filename='all-reg.jpg'):
        index = 0
        for xgb_reg_obj in self.xg_regs:
            curr_model = self.conversions[index]
            curr_model = MapfEDA.folder_from_label(curr_model)
            xgb_obj = xgb_reg_obj.regressor_
            explainer = shap.TreeExplainer(xgb_obj)
            shap_values = explainer.shap_values(df[self.features_cols])
            plt.figure(figsize=(15, 15))

            shap.summary_plot(shap_values, df[self.features_cols], show=False)
            plt.savefig("SHAP/" + curr_model + '-' + filename, bbox_inches="tight")
            plt.close('all')
            index += 1
