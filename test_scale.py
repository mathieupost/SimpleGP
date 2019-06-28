import random
import sys

from tqdm import tqdm

from cross_validation import CrossValidation
from logger.multi_logger import MultiLogger
from simplegp.Evolution.Evolution import SimpleGP
# Internal imports
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Weights.Tuner import Tuner

np.random.seed(42)
random.seed(42)

settings = [
    # Normal
    (False, Tuner(), "normal"),
    # Normal w LS
    (True, Tuner(), "normal_ls"),
    # # Tuner in all generations
    # (
    #     False,
    #     Tuner(
    #         scale_range=(-5, 5),
    #         translation_range=(-5, 5),
    #         run_generations=(range(0, 100))),
    #     "tuner"
    # ),
    # # Tuner in all generations w LS
    # (
    #     True,
    #     Tuner(
    #         scale_range=(-5, 5),
    #         translation_range=(-5, 5),
    #         run_generations=(range(0, 100))),
    #     "tuner_ls")
]


def run_with_settings(ls, tuner):
    # Set functions and terminals
    functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
    terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node

    # Run GP
    sgp = SimpleGP(linear_scale=ls, tuner=tuner, functions=functions, pop_size=500, max_generations=100)

    CrossValidation(sgp, terminals).validate()


if __name__ == '__main__':
    for setting in tqdm(settings, desc="Test Linear Scaling"):
        ls, tuner, name = setting
        with open(f"log/log_scale_{name}.txt", 'w+') as logfile:
            mc = MultiLogger([sys.stdout, logfile])
            mc.capture(run_with_settings, ls, tuner)
