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
    # Thieu
    # Run every 5 generations (gen starts with 0, so gen 4 is actually the 5th generation)
    (False, range(4, 100, 5), "tune_5_gen_max_gen_100"),

    # Kasper
    # Run every 5 generations with ls
    (True, range(4, 100, 5), "tune_5_gen_ls_max_gen_100"),

    # Timo
    # Run every 20 generations
    (False, range(19, 100, 20), "tune_20_gen_max_gen_100"),

    # Sven
    # Run every 20 generations with ls
    (False, range(19, 100, 20), "tune_20_gen_ls_max_gen_100")
]


def run_in_gen(ls, run_gen):
    # Set functions and terminals
    functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
    terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node

    # Run GP
    tuner = Tuner(
        scale_range=(-5, 5),
        translation_range=(-5, 5),
        run_generations=(run_gen)
    )
    sgp = SimpleGP(
        linear_scale=ls,
        tuner=tuner,
        functions=functions,
        pop_size=100,
        max_generations=100
    )

    CrossValidation(sgp, terminals).validate()


if __name__ == '__main__':
    for setting in tqdm(settings, desc="Test Weights"):
        ls, run_gen, name = setting
        with open(f"log/log_{name}.txt", "w+") as logfile:
            mc = MultiLogger([sys.stdout, logfile])
            mc.capture(run_in_gen, ls, run_gen)
