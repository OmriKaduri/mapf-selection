import csv
import os

import hyperopt
import numpy as np
import pandas as pd

from hyperopt import space_eval

from HyperoptObjective import HyperoptObjective


class MapfModel:
    def __init__(self, runtime_cols, max_runtime, features_cols, success_cols=None, maptype=''):
        self.runtime_cols = runtime_cols
        self.max_runtime = max_runtime
        self.features_cols = features_cols
        self.success_cols = success_cols
        self.alg_runtime_cols = self.runtime_cols.copy()
        self.only_alg_runtime_cols = runtime_cols.copy()
        self.only_alg_runtime_cols.remove('Y Runtime')
        # self.train_samples_weight = X_train.apply(lambda x:
        #                                           np.log1p(np.std(x[self.only_alg_runtime_cols].values)), axis=1)
        self.modelname = 'model'
        self.maptype = maptype
        self.conversions = dict(zip(np.arange(len(self.only_alg_runtime_cols)), iter(self.only_alg_runtime_cols)))
        self.results = pd.DataFrame(
            columns=['Model', 'Accuracy', 'Coverage', 'Cumsum', 'Normalized Coverage'])

    def find_best_params(self,
                         X_train,
                         y_train,
                         X_test,
                         y_test,
                         model,
                         const_params,
                         parameter_space,
                         fit_params={},
                         max_evals=25,
                         is_regressor=False,
                         is_binary=True
                         ):
        objective = HyperoptObjective(X_train, y_train, X_test, y_test, model, const_params,
                                      fit_params, self.features_cols, is_regressor, is_binary)
        '''
        HyperOpt Trials object stores details of every iteration. 
        https://github.com/hyperopt/hyperopt/wiki/FMin#12-attaching-extra-information-via-the-trials-object
        '''
        trials = hyperopt.Trials()

        '''
        Hyperopt fmin function returns only parameters from the search space.
        Therefore, before returning best_params
        we will merge best_params with the const params, 
        so we have all parameters in one place for training the best model.
        '''
        best_params = hyperopt.fmin(
            fn=objective,
            space=parameter_space,
            algo=hyperopt.tpe.suggest,
            rstate=np.random.RandomState(seed=42),
            max_evals=max_evals,
            trials=trials
        )

        best_params = space_eval(parameter_space, best_params)
        best_params.update(const_params)
        return best_params, trials

    def add_modelname_suffix(self, suffix):
        self.modelname += '-' + suffix

    def sample_weight(self, X_train):
        # The sample weight is 2^(#Algs-#AlgsSolvedProblem)
        # return 2 ** (len(self.only_alg_runtime_cols) - X_train[self.success_cols].sum(axis=1))
        return len(self.only_alg_runtime_cols) - X_train[self.success_cols].sum(axis=1) + 1
        # return [1] * len(X_train)

    def class_weight(self, X_train, y_train):
        total = len(X_train)
        class_weights = {}
        for alg in self.only_alg_runtime_cols:
            n_samples = len(X_train[X_train['Y'] == alg])
            class_weights[alg] = n_samples / total
        sample_weights = [class_weights[self.only_alg_runtime_cols[y]] for y in y_train]
        return sample_weights

    def print_results(self, results_file='model-results.csv', with_header=True, exp_dir=""):
        if self.results.empty:
            return
        # fieldnames = ['Model', 'Accuracy', 'Coverage', 'Cumsum(minutes)', 'Normalized Coverage', 'Normalized Accuracy']
        fieldnames = ['Model', 'Cumsum(minutes)', 'Normalized Coverage', 'Normalized Accuracy']
        results_df = pd.DataFrame(columns=fieldnames)
        if exp_dir != '':
            exp_dir = 'exp/' + exp_dir
            os.makedirs(exp_dir, exist_ok=True)
        with open(exp_dir + results_file, 'a+', newline='') as csvfile:
            res_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if with_header:
                res_writer.writeheader()

            metrics_mean = self.results.groupby('Model').mean()
            if self.results.groupby('Model').size().max() == 1:
                metrics_std = self.results.groupby('Model').mean()
            else:
                metrics_std = self.results.groupby('Model').std()
            for model in self.results['Model'].unique():
                if self.maptype != '':
                    modelname = model + '-' + self.maptype
                else:
                    modelname = model
                with_std = False
                if with_std:
                    model_results = {'Model': modelname,
                                     # 'Accuracy': "{0:.2%}{1}{2:.2%}".format(
                                     #     metrics_mean['Accuracy'][model], u"\u00B1", metrics_std['Accuracy'][model]),
                                     # 'Coverage': "{0:.2%}{1}{2:.2%}".format(
                                     #     metrics_mean['Coverage'][model], u"\u00B1", metrics_std['Coverage'][model]),
                                     'Cumsum(minutes)': "{0}{1}{2}".format(
                                         int(metrics_mean['Cumsum'][model]), u"\u00B1",
                                         int(metrics_std['Cumsum'][model])),
                                     'Normalized Coverage': "{0:.2%}{1}{2:.2%}".format(
                                         metrics_mean['Normalized Coverage'][model], u"\u00B1",
                                         metrics_std['Normalized Coverage'][model]),
                                     'Normalized Accuracy': "{0:.2%}{1}{2:.2%}".format(
                                         metrics_mean['Normalized Accuracy'][model], u"\u00B1",
                                         metrics_std['Normalized Accuracy'][model]),
                                     }
                else:
                    model_results = {'Model': modelname,
                                     # 'Accuracy': "{0:.2f}".format(100 * metrics_mean['Accuracy'][model]),
                                     # 'Coverage': "{0:.2f}".format(100 * metrics_mean['Coverage'][model]),
                                     'Cumsum(minutes)': "{0}".format(int(metrics_mean['Cumsum'][model])),
                                     'Normalized Coverage': "{0:.2f}".format(
                                         100 * metrics_mean['Normalized Coverage'][model]),
                                     'Normalized Accuracy': "{0:.2f}".format(
                                         100 * metrics_mean['Normalized Accuracy'][model])
                                     }

                model_results_for_latex = {'Model': modelname,
                                           # 'Accuracy': "{0:.2f}".format(metrics_mean['Accuracy'][model] * 100),
                                           # 'Coverage': "{0:.2f}".format(metrics_mean['Coverage'][model] * 100),
                                           'Cumsum(minutes)': "{0}".format(int(metrics_mean['Cumsum'][model])),
                                           'Normalized Coverage': "{0:.2%}".format(
                                               metrics_mean['Normalized Coverage'][model] * 100),
                                           'Normalized Accuracy': "{0:.2%}".format(
                                               metrics_mean['Normalized Accuracy'][model] * 100),
                                           }
                results_df = results_df.append(model_results_for_latex, ignore_index=True)
                res_writer.writerow(model_results)

        return results_df
