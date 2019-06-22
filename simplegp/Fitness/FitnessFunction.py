from copy import deepcopy

import numpy as np


class SymbolicRegressionFitness:

    def __init__(self, X_train, y_train, linear_scale):
        self.X_train = X_train
        self.y_train = y_train
        self.elite = None
        self.evaluations = 0
        self.linear_scale = linear_scale

        # precompute data mean as it wont change
        if linear_scale:
            self.y_mean = np.mean(y_train)

    def _mse(self, y, output, test=False):
        a = 0
        b = 1
        if self.linear_scale:
            # Compute a and b on the available training data and NEVER on the testing data
            if test:
                # We need to compute the scaling factors, but have no information about the targets y
                # Therefore, we make predictions with the training data and compute a & b
                t_output = self.elite.get_output(self.X_train)
                b = np.cov(self.y_train, t_output)[1][0] / np.var(t_output)
                a = self.y_mean - b * np.mean(t_output)
            else:
                var = np.var(output)
                if var == 0:
                    # No variance, so we we have a tree with one an emperical constant node (a straight line)
                    # We only compute a and scale as this also gives the minimal error
                    # This only happens during training with small trees
                    b = 1
                else:
                    b = np.cov(self.y_train, output)[1][0] / var

                a = self.y_mean - b * np.mean(output)

        # Compute MSE
        return np.mean(np.square(y - (a + b * output)))

    def evaluate(self, individual):
        self.evaluations = self.evaluations + 1

        output = individual.get_output(self.X_train)
        individual.fitness = self._mse(self.y_train, output)

        if not self.elite or individual.fitness < self.elite.fitness:
            del self.elite
            self.elite = deepcopy(individual)

    def test(self, X_test, y_test):
        # Test with the elite
        output = self.elite.get_output(X_test)
        return self._mse(y_test, output, test=True)
