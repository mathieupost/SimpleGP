from copy import deepcopy
from random import random

import numpy as np
from gaft.plugin_interfaces.operators.crossover import Crossover


class GaussianCrossover(Crossover):
    ''' Crossover operator for real-valued individuals

    :param pc: The probability of crossover (usually between 0.25 ~ 1.0)
    :type pc: float in (0.0, 1.0]
    '''

    def __init__(self, pc):
        if pc <= 0.0 or pc > 1.0:
            raise ValueError('Invalid crossover probability')
        self.pc = pc

    def _get_stats(self, chrom1, chrom2):
        # Compute mean and std per dimension
        mean = np.mean((chrom1, chrom2), axis=0)
        std = np.std((chrom1, chrom2), axis=0)

        return mean, std

    def _multivariate(self, chrom1, chrom2):
        mean, std = self._get_stats(chrom1, chrom2)

        # Create covariance matrix with all covariance terms zeros
        # Thus only set the diagonal with std^2
        cov = np.zeros((len(chrom1), len(chrom1)))
        row, col = np.diag_indices_from(cov)
        cov[row, col] = np.power(std, 2)

        # Sample from this distribution
        samples = np.random.multivariate_normal(mean, cov, size=2)

        return samples

    def _gaussian(self, chrom1, chrom2):
        mean, std = self._get_stats(chrom1, chrom2)

        length = len(chrom1)
        child1 = np.zeros(length)
        child2 = np.zeros(length)

        # Sample each dimension separately
        for dim in range(len(chrom1)):
            s = np.random.normal(mean[dim], std[dim], size=2)

            child1[dim] = s[0]
            child2[dim] = s[1]

        return [child1, child2]

    def cross(self, father, mother):
        ''' Cross chromosomes of parent using gaussian crossover method.

        :param population: Population where the selection operation occurs.
        :type population: :obj:`gaft.components.Population`

        :return: Selected parents (a father and a mother)
        :rtype: list of :obj:`gaft.components.IndividualBase`
        '''

        do_cross = True if random() <= self.pc else False

        if not do_cross:
            return father.clone(), mother.clone()

        # Chromsomes for two children
        chrom1 = deepcopy(father.chromsome)
        chrom2 = deepcopy(mother.chromsome)

        if len(chrom1) != len(chrom2):
            raise ValueError("chromosomes unequal length!")

        # Gaussian is used as it seems to be faster to sample each dimension separately
        # samples = self._multivariate(chrom1, chrom2)
        samples = self._gaussian(chrom1, chrom2)

        # Create the childs from the samples
        child1, child2 = father.clone(), father.clone()
        child1.init(chromsome=list(samples[0]))
        child2.init(chromsome=list(samples[1]))

        return child1, child2
