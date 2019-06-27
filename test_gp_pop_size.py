# Libraries

import random
import sys

from tqdm import tqdm

from cross_validation import CrossValidation
from logger.multi_logger import MultiLogger
from simplegp.Evolution.Evolution import SimpleGP
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Weights.Tuner import Tuner

np.random.seed(42)
random.seed(42)

settings = [10]


def run_with_population(pop_size):
    # Set functions and terminals
    functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
    terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node

    # Run GP
    tuner = Tuner()
    sgp = SimpleGP(tuner=tuner, functions=functions, pop_size=pop_size, max_generations=100)

    CrossValidation(sgp, terminals).validate()


if __name__ == '__main__':
    for pop in tqdm(settings, desc="Test Population Size"):
        with open(f"log/log_pop_size_{pop}.txt", "w+") as logfile:
            mc = MultiLogger([sys.stdout, logfile])
            mc.capture(run_with_population, pop)
