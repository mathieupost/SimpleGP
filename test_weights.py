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


def run_with_range(range):
    # Set functions and terminals
    functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
    terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node

    # Run GP
    tuner = Tuner(
        scale_range=(range[0], range[1]),
        translation_range=(range[0], range[1]),
        run_generations=()
    )
    sgp = SimpleGP(tuner=tuner, functions=functions, pop_size=100, max_generations=100)

    CrossValidation(sgp, terminals).validate()


for setting in tqdm(settings, desc="Test Weights"):
    with open(f"log/log_weight_{setting[0]}_{setting[1]}.txt", "a") as logfile:
        mc = MultiLogger([sys.stdout, logfile])
        mc.capture(run_with_range, setting)
