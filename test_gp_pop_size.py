# Libraries

import random
import sys

from cross_validation import CrossValidation
from simplegp.Evolution.Evolution import SimpleGP
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Weights.Tuner import Tuner

np.random.seed(42)
random.seed(42)

settings = [2, 10, 100, 500, 1000, 2000]

for pop in settings:
    sys.stdout = open(f"log/log_pop_size_{pop}.txt", "w+")
    # Set functions and terminals
    functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
    terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node

    # Run GP
    tuner = Tuner()
    sgp = SimpleGP(tuner=tuner, functions=functions, pop_size=pop, max_generations=100)  # other parameters are optional

    CrossValidation(sgp, terminals).validate()
