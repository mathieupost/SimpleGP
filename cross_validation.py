import copy
import multiprocessing as mp
import os
import sys
from io import StringIO

import numpy as np
import platform
import psutil
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from logger.multi_logger import MultiLogger
from logger.prefix_logger import PrefixLogger
from simplegp.Evolution.Evolution import SimpleGP
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Nodes.SymbolicRegressionNodes import FeatureNode


class CrossValidation:
    def __init__(self, GA: SimpleGP, terminals, X=None, y=None, ksplits=10):
        self.GA = GA

        # Set the default Boston dataset
        if X is None and y is None:
            X, y = datasets.load_boston(return_X_y=True)
            self.X = X
            self.y = y
        else:
            self.X = X
            self.y = y

        self.ksplits = ksplits

        # Set the feature terminals as it is based on the size of the dataset
        for i in range(X.shape[1]):
            terminals.append(FeatureNode(i))  # add a feature node for each feature

        self.GA.terminals = terminals

    def get_splits(self):
        kf = KFold(n_splits=self.ksplits, random_state=42)
        X = self.X
        y = self.y

        for trn_idx, tst_idx in kf.split(X):
            # Get the data for the split
            X_train, X_test = X[trn_idx], X[tst_idx]
            y_train, y_test = y[trn_idx], y[tst_idx]

            # Fit the scaler on the training set
            X_scaler = StandardScaler()
            # y_scaler = StandardScaler()

            X_train = X_scaler.fit_transform(X_train)
            # y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))

            # Transform the test data
            X_test = X_scaler.transform(X_test, copy=True)
            # y_test = y_scaler.transform(y_test.reshape(-1, 1), copy=True)
            # TODO: we dont scale the targets

            yield X_train, X_test, y_train, y_test

    def validate_split(self, split):
        #  Set process to low priority
        p = psutil.Process(os.getpid())
        if platform.system() == 'Windows':
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(19)

        run, X_train, X_test, y_train, y_test = split

        def f():
            print(f'Run{run}')
            ga = copy.deepcopy(self.GA)
            # Set the data in the fitness function
            fun = SymbolicRegressionFitness(X_train, y_train, ga.linear_scale)
            ga.fitness_function = fun
            ga.tuner.fitness_function = fun

            # Run the GA and get the best function
            ga.run()
            best_function = ga.fitness_function.elite
            best_fitness = best_function.fitness

            # Compute the training metrics
            train_mse = best_fitness
            train_R = 1.0 - best_fitness / np.var(y_train)

            # Make predictions with the elite and get the MSE
            test_mse = ga.fitness_function.test(X_test, y_test)
            test_R = 1.0 - test_mse / np.var(y_test)

            print(f'KFold Run: {run}\n'
                  'Training\n'
                  f'\tMSE: {np.round(train_mse, 3)}\n'
                  f'\tRsquared: {np.round(train_R, 3)}\n'
                  'Testing\n'
                  f'\tMSE: {np.round(test_mse, 3)}\n'
                  f'\tRsquared: {np.round(test_R, 3)}')

            return train_mse, train_R, test_mse, test_R

        log = StringIO()
        stdout_with_prefix = PrefixLogger(sys.__stdout__, prefix=f"Run{run}")
        mc = MultiLogger([stdout_with_prefix, log])
        return (*mc.capture(f), log.getvalue())

    def validate(self):
        train_mses = []
        train_Rs = []

        test_mses = []
        test_Rs = []

        with mp.Pool(processes=self.ksplits) as pool:
            splits = []
            run = 0
            for X_train, X_test, y_train, y_test in self.get_splits():
                splits.append([run, X_train, X_test, y_train, y_test])
                run += 1

            results = pool.map(self.validate_split, splits)
            for res in results:
                train_mse, train_R, test_mse, test_R, log = res

                train_mses.append(train_mse)
                train_Rs.append(train_R)

                test_mses.append(test_mse)
                test_Rs.append(test_R)

                print(log)

        print('KFold Result\n'
              'Testing\n'
              f'\tMSE: {np.mean(test_mses)} {np.std(test_mses)}\n'
              'Training\n'
              f'\tMSE: {np.mean(train_mses)} {np.std(train_mses)}')

        # Return the average and the deviation over the runs
        return (np.mean(test_mses), np.std(test_mses)), (np.mean(test_Rs), np.std(test_Rs))
