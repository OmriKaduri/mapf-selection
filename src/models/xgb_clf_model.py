import os
from pathlib import Path
import hyperopt
from metrics import normalized_coverage_score, normalized_accuracy_score, cumsum_score, runtime_adjusted_coverage_score
from models.mapf_model import MapfModel
import xgboost as xgb
import numpy as np
from preprocess import Preprocess
from sklearn.model_selection import GroupShuffleSplit
import shap
import matplotlib.pyplot as plt
from mapf_eda import MapfEDA
from joblib import dump, load


class XGBClfModel(MapfModel):

    def __init__(self, *args):
        super(XGBClfModel, self).__init__(*args)
        self.xg_cls = {}
        self.balanced = False
        self.modelname = 'XGBoost Classification'
        if self.maptype != '':
            self.modelname += '-' + self.maptype

    def balance_dataset(self):
        self.balanced = True
        self.modelname += ' - balanced'

    def train_cv(self, data, labels, n_splits=2, hyperopt_evals=5, load=False, model_suffix='clf-model.xgb',
                 models_dir='models/classification', exp_type='in-map-300000'):
        if self.balanced:
            data = Preprocess.balance_dataset_by_label(data)
            labels = data['Y']

        if len(set(data['InstanceId'])) > 1:
            groups = data['InstanceId']  # len of scenarios
        else:
            groups = data.index

        gkf = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
        if len(set(labels)) == 2:
            objective = 'binary:logistic'
        else:
            objective = 'multi:softmax'
        self.xg_cls = xgb.XGBClassifier(objective=objective)
        model_dir = Path(models_dir) / exp_type
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(model_dir / model_suffix)

        if load and os.path.exists(model_path):
            print("loaded classification model from", model_path)
            self.xg_cls.load_model(model_path)
            return

        param_dist = {'n_estimators': hyperopt.hp.choice('n_estimators', np.arange(75, 500, 25)),
                      'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 0.07),
                      'subsample': hyperopt.hp.uniform('subsample', 0.3, 0.7),
                      'max_depth': hyperopt.hp.choice('max_depth', np.arange(3, 10)),
                      'min_child_weight': hyperopt.hp.choice('min_child_weight', np.arange(1, 4)),
                      "gamma": hyperopt.hp.choice('gamma', np.arange(0.1, 2, 0.2)),
                      "reg_alpha": hyperopt.hp.choice('reg_alpha', np.arange(0, 1.5, 0.5))
                      }

        best_coverage = 0
        best_params = {}
        for index, (tr_ind, test_ind) in enumerate(gkf.split(data[self.features_cols], labels, groups)):
            print("Starting {i} inner fold out of {n} in classification training".format(i=index, n=n_splits))
            X_train = data.iloc[tr_ind].copy()
            y_train = labels.iloc[tr_ind].copy()

            X_test = data.iloc[test_ind].copy()
            y_test = labels.iloc[test_ind].copy()
            curr_best_params, trials = self.find_best_params(X_train, y_train, X_test, y_test,
                                                             xgb.XGBClassifier,
                                                             {'objective': objective},
                                                             param_dist,
                                                             {
                                                                 'sample_weight': self.sample_weight(X_train),
                                                                 # 'sample_weight': self.class_weight(X_train, y_train)
                                                             },
                                                             max_evals=hyperopt_evals,
                                                             is_binary=True)

            fnvals = [(t['result']) for t in trials.trials]
            params = max(fnvals, key=lambda x: -x['loss'])
            if -params['loss'] > best_coverage:
                best_coverage = -params['loss']
                best_params = curr_best_params

        print("Best params", best_params)
        self.xg_cls = xgb.XGBClassifier(**best_params)
        self.xg_cls = self.xg_cls.fit(data[self.features_cols], labels,
                                      sample_weight=self.sample_weight(data),
                                      # sample_weight=self.class_weight(data, labels)
                                      )
        self.xg_cls.save_model(model_path)

    def predict(self, X_test, y_test, online_feature_extraction_time=None):
        test_preds = self.xg_cls.predict(X_test[self.features_cols])
        test_preds = [self.conversions[index] for index in test_preds]

        model_acc = normalized_accuracy_score(X_test, test_preds)
        if online_feature_extraction_time:
            model_coverage = runtime_adjusted_coverage_score(X_test, test_preds,
                                                             (self.max_runtime - X_test[
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

    def plot_feature_importance(self, df, filename='all-cls.jpg'):
        # shap.initjs()
        print("Plot classification feature importance for", filename)
        explainer = shap.TreeExplainer(self.xg_cls)
        shap_values = explainer.shap_values(df[self.features_cols])
        plt.figure(figsize=(15, 15))
        shap.summary_plot(shap_values, df[self.features_cols],
                          show=False,
                          class_names=[MapfEDA.model_name_for_plot(c) for c in self.xg_cls.classes_])
        plt.savefig("SHAP/" + filename, bbox_inches="tight")
        plt.close('all')

    @staticmethod
    def cov_scorer(y, y_pred, **kwargs):
        max_runtime = kwargs['max_runtime']  # should be 300000 by default
        df = kwargs['df']  # should be the df with runtime_cols
        tmp_df = df.copy()
        print(y_pred)
        tmp_df['CurrP'] = y_pred
        tmp_df['CurrP-Runtime'] = tmp_df.apply(lambda x: x[x['CurrP']], axis=1)
        solved = len(tmp_df[tmp_df['CurrP-Runtime'] < max_runtime])
        return solved / len(tmp_df)
