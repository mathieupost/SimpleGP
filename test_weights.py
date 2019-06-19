# Libraries

import random

from cross_validation import CrossValidation
from simplegp.Evolution.Evolution import SimpleGP
# Internal imports
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Weights.Tuner import Tuner
import sys

np.random.seed(42)
random.seed(42)

settings = [
    [-1, 1],
    [-3, 3],
    [-5, 5],
    [-10, 10],
    [0, 1],
    [0, 3],
    [0, 5],
    [0, 10],
]

for setting in settings:
    sys.stdout = open('log/log_weight_{}_{}.txt'.format(setting[0], setting[1]), 'a')

    # Set functions and terminals
    functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
    terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node

    # Run GP
    tuner = Tuner(
        scale_range=(setting[0], setting[1]),
        translation_range=(setting[0], setting[1]),
        run_generations=()
    )
    sgp = SimpleGP(tuner=tuner, functions=functions, pop_size=100, max_generations=100)

    CrossValidation(sgp, terminals).validate()
