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

settings = [10, 20, 50, 100, 500, 1000]


def run_with_range(range):
    # Set functions and terminals
    functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
    terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node

    # Run GP
    tuner = Tuner(
        pop_size=range,
        run_generations=(99, 100)
    )
    sgp = SimpleGP(tuner=tuner, functions=functions, pop_size=100, max_generations=100)

    CrossValidation(sgp, terminals).validate()


if __name__ == '__main__':
    for setting in tqdm(settings, desc="Test Weights"):
        with open(f"log/log_tuner_pop_size_{setting}.txt", "w+") as logfile:
            mc = MultiLogger([sys.stdout, logfile])
            mc.capture(run_with_range, setting)
