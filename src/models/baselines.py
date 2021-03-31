from models.mapf_model import MapfModel
import numpy as np
from sklearn.metrics import accuracy_score
from metrics import coverage_score, cumsum_score, normalized_coverage_score, normalized_accuracy_score


class Baselines(MapfModel):

    def _compute_best_solver_metrics(self, X_train, X_test, y_test, max_runtime, by_grid=True, by_coverage=True,
                                     by_normalized_coverage=False, by_normalized_accuracy=False):
        groupby_key = 'GridName'
        model_name = 'Best-at-grid'
        if not by_grid:
            groupby_key = 'maptype'
            model_name = 'Best-at-maptype'

        if by_normalized_coverage:
            model_name += '-by-norm-cov'
        elif by_normalized_accuracy:
            model_name += '-by-norm-acc'
        else:
            if not by_coverage:
                model_name += '-by-acc'

        grids = X_train.groupby(groupby_key)
        grid_results = {}
        grid_solvers = {}
        for grid, group in grids.groups.items():
            grid_results[grid] = 0  # Minimum coverage init
            for solver in self.conversions.values():
                static_alg_selection = [solver] * len(group)
                if by_normalized_coverage:
                    result = normalized_coverage_score(X_train.loc[group], static_alg_selection, max_runtime)
                elif by_normalized_accuracy:
                    result = normalized_accuracy_score(X_train.loc[group], static_alg_selection)
                else:
                    if by_coverage:
                        result = coverage_score(X_train.loc[group], static_alg_selection, max_runtime)
                    else:
                        result = accuracy_score(X_train.loc[group]['Y'], static_alg_selection)
                if result > grid_results[grid]:  # Better coverage/accuracy
                    grid_results[grid] = result
                    grid_solvers[grid] = solver  # Set the best solver at grid

        print(grid_results)
        print("Mean:", np.mean(list(grid_results.values())))
        print("Std:", np.std(list(grid_results.values())))
        # print(grid_solvers)
        test_groups = X_test.copy().groupby(groupby_key)
        for t_grid, t_group in test_groups.groups.items():
            if by_normalized_coverage:
                new_result = normalized_coverage_score(X_test.copy().loc[t_group], grid_solvers.copy()[t_grid],
                                                       max_runtime)
            else:
                new_result = normalized_accuracy_score(X_test.copy().loc[t_group],
                                                       [grid_solvers.copy()[t_grid]] * len(t_group))
            print("Grid:", t_grid, "Result:", new_result)

        preds = X_test.apply(lambda x: Baselines.pick_best_solver_at_grid(grid_solvers, x, groupby_key), axis=1)
        # best_at_grid_acc = accuracy_score(y_test, preds)
        # best_at_grid_coverage = coverage_score(X_test, preds, max_runtime)
        best_at_grid_cumsum = cumsum_score(X_test, preds)
        best_at_grid_normalized_coverage = normalized_coverage_score(X_test, preds, max_runtime)
        best_at_grid_normalized_acc = normalized_accuracy_score(X_test, preds)
        # print("{m} Coverage:".format(m=model_name), best_at_grid_coverage)
        # print("{m} accuracy:".format(m=model_name), best_at_grid_acc)
        print("{m} cumsum:".format(m=model_name), best_at_grid_cumsum)
        print("{m} normalized coverage:".format(m=model_name), best_at_grid_normalized_coverage)
        print("{m} normalized accuracy:".format(m=model_name), best_at_grid_normalized_acc)

        return {'Model': '{m} baseline'.format(m=model_name),
                # 'Accuracy': best_at_grid_acc,
                # 'Coverage': best_at_grid_coverage,
                'Cumsum': best_at_grid_cumsum,
                'Normalized Coverage': best_at_grid_normalized_coverage,
                'Normalized Accuracy': best_at_grid_normalized_acc}, preds

    @staticmethod
    def pick_best_solver_at_grid(grid_solvers, row, key):
        return grid_solvers[row[key]]

    def predict(self, X_train, X_test, y_test):
        baseline_preds = {}
        random_preds = [self.conversions[x] for x in
                        np.random.randint(0, len(self.only_alg_runtime_cols), size=(len(X_test)))]
        # random_acc = accuracy_score(y_test, random_preds)
        # random_coverage = coverage_score(X_test, random_preds, self.max_runtime)
        random_cumsum = cumsum_score(X_test, random_preds)
        random_normalized_coverage = normalized_coverage_score(X_test, random_preds, self.max_runtime)
        random_normalized_accuracy = normalized_accuracy_score(X_test, random_preds)
        # print("Random baseline coverage:", random_coverage)
        # print("Random baseline accuracy:", random_acc)
        print("Random baseline cumsum:", random_cumsum)
        print("Random baseline normalized coverage:", random_normalized_coverage)
        print("Random baseline normalized accuracy:", random_normalized_accuracy)
        self.results = self.results.append({'Model': "Random baseline",
                                            # 'Accuracy': random_acc,
                                            # 'Coverage': random_coverage,
                                            'Cumsum': random_cumsum,
                                            'Normalized Coverage': random_normalized_coverage,
                                            'Normalized Accuracy': random_normalized_accuracy}
                                           , ignore_index=True)
        for key, conversion in self.conversions.items():
            preds = [conversion] * len(X_test)
            # alg_acc = accuracy_score(y_test, preds)
            # alg_coverage = coverage_score(X_test, preds, self.max_runtime)
            alg_cumsum = cumsum_score(X_test, preds)
            alg_normalized_coverage = normalized_coverage_score(X_test, preds, self.max_runtime)
            alg_normalized_accuracy = normalized_accuracy_score(X_test, preds)
            self.results = self.results.append({
                'Model': conversion,
                # 'Accuracy': alg_acc,
                # 'Coverage': alg_coverage,
                'Cumsum': alg_cumsum,
                'Normalized Coverage': alg_normalized_coverage,
                'Normalized Accuracy': alg_normalized_accuracy
            }, ignore_index=True)
            # print("{m} baseline coverage:".format(m=conversion), alg_coverage)
            # print("{m} baseline accuracy:".format(m=conversion), alg_acc)
            print("{m} baseline cumsum:".format(m=conversion), alg_cumsum)
            print("{m} baseline normalized coverage:".format(m=conversion), alg_normalized_coverage)
            print("{m} baseline normalized accuracy:".format(m=conversion), alg_normalized_accuracy)

        if set(X_test['GridName']) != set(X_train['GridName']):
            print("Some maps from test wasn't found at train, thus couldn't compute best-at-grid baseline.")
        else:
            # best_at_grid_results, best_at_grid_preds = self._compute_best_solver_metrics(X_train, X_test, y_test,
            #                                                                              self.max_runtime,
            #                                                                              by_grid=True,
            #                                                                              by_coverage=True)
            # best_at_grid_acc_results, best_at_grid_acc_preds = self._compute_best_solver_metrics(X_train, X_test,
            #                                                                                      y_test,
            #                                                                                      self.max_runtime,
            #                                                                                      by_grid=True,
            #                                                                                      by_coverage=False)
            best_at_grid_by_norm_results, best_at_grid_by_norm_preds = self._compute_best_solver_metrics(X_train,
                                                                                                         X_test,
                                                                                                         y_test,
                                                                                                         self.max_runtime,
                                                                                                         by_grid=True,
                                                                                                         by_coverage=False,
                                                                                                         by_normalized_coverage=True)

            best_at_grid_by_norm_acc_results, best_at_grid_by_norm_acc_preds = self._compute_best_solver_metrics(
                X_train,
                X_test,
                y_test,
                self.max_runtime,
                by_grid=True,
                by_coverage=False,
                by_normalized_coverage=False,
                by_normalized_accuracy=True)

            # self.results = self.results.append(best_at_grid_results, ignore_index=True)
            # baseline_preds['P-Best-at-grid'] = best_at_grid_preds
            # self.results = self.results.append(best_at_grid_acc_results, ignore_index=True)
            # baseline_preds['P-Best-at-grid-by-acc'] = best_at_grid_acc_preds
            self.results = self.results.append(best_at_grid_by_norm_results, ignore_index=True)
            baseline_preds['P-Best-at-grid-by-norm'] = best_at_grid_by_norm_preds
            self.results = self.results.append(best_at_grid_by_norm_acc_results, ignore_index=True)
            baseline_preds['P-Best-at-grid-by-norm-acc'] = best_at_grid_by_norm_acc_results

        if set(X_test['maptype']) != set(X_train['maptype']) or len(set(X_test['maptype'])) == 1:
            print("Some map TYPES from test wasn't found at train, thus couldn't compute best-at-type baseline.")
        else:
            # best_at_type_results, best_at_type_preds = self._compute_best_solver_metrics(X_train, X_test, y_test,
            #                                                                              self.max_runtime,
            #                                                                              by_grid=False,
            #                                                                              by_coverage=True)
            # best_at_type_acc_results, best_at_type_acc_preds = self._compute_best_solver_metrics(X_train, X_test,
            #                                                                                      y_test,
            #                                                                                      self.max_runtime,
            #                                                                                      by_grid=False,
            #                                                                                      by_coverage=False)
            best_at_type_by_norm_results, best_at_type_by_norm_preds = self._compute_best_solver_metrics(X_train,
                                                                                                         X_test,
                                                                                                         y_test,
                                                                                                         self.max_runtime,
                                                                                                         by_grid=False,
                                                                                                         by_coverage=False,
                                                                                                         by_normalized_coverage=True)
            best_at_type_by_norm_acc_results, best_at_type_by_norm_acc_preds = self._compute_best_solver_metrics(
                X_train,
                X_test,
                y_test,
                self.max_runtime,
                by_grid=False,
                by_coverage=False,
                by_normalized_coverage=False,
                by_normalized_accuracy=True)

            # self.results = self.results.append(best_at_type_results, ignore_index=True)
            # baseline_preds['P-Best-at-maptype'] = best_at_type_preds
            # self.results = self.results.append(best_at_type_acc_results, ignore_index=True)
            # baseline_preds['P-Best-at-maptype-by-acc'] = best_at_type_acc_preds
            self.results = self.results.append(best_at_type_by_norm_results, ignore_index=True)
            baseline_preds['P-Best-at-maptype-by-norm'] = best_at_type_by_norm_preds
            self.results = self.results.append(best_at_type_by_norm_acc_results, ignore_index=True)
            baseline_preds['P-Best-at-maptype-by-norm-acc'] = best_at_type_by_norm_acc_preds

        # optimal_acc = accuracy_score(y_test, X_test['Y'])
        # optimal_coverage = coverage_score(X_test, X_test['Y'], self.max_runtime)
        optimal_cumsum = cumsum_score(X_test, X_test['Y'])
        optimal_normalized_coverage = normalized_coverage_score(X_test, X_test['Y'], self.max_runtime)
        optimal_normalized_accuracy = normalized_accuracy_score(X_test, X_test['Y'])

        self.results = self.results.append({'Model': 'Optimal Oracle',
                                            # 'Accuracy': optimal_acc,
                                            # 'Coverage': optimal_coverage,
                                            'Cumsum': optimal_cumsum,
                                            'Normalized Coverage': optimal_normalized_coverage,
                                            'Normalized Accuracy': optimal_normalized_accuracy}
                                           , ignore_index=True)
        # print("Oracle baseline coverage:", optimal_coverage)
        # print("Oracle baseline accuracy:", optimal_acc)
        print("Oracle baseline cumsum:", optimal_cumsum)
        print("Oracle baseline normalize coverage:", optimal_normalized_coverage)
        print("Oracle baseline normalize accuracy:", optimal_normalized_accuracy)

        return baseline_preds
