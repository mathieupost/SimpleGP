import random
import sys
import numpy as np

from cross_validation import CrossValidation
from simplegp.Evolution.Evolution import SimpleGP
# Internal imports
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Weights.Tuner import Tuner

from tqdm import tqdm

np.random.seed(42)
random.seed(42)

settings = [
    # Normal
    # (False, Tuner(), "normal"),
    # Normal w LS
    (True, Tuner(), "normal_ls"),
    # Tuner
    # (
    #     False,
    #     Tuner(
    #         scale_range=(-5, 5),
    #         translation_range=(-5, 5),
    #         run_generations=()),
    #     "tuner"
    # ),
    # # Tuner w LS
    # (
    #     True,
    #     Tuner(
    #         scale_range=(-5, 5),
    #         translation_range=(-5, 5),
    #         run_generations=()),
    #     "tuner_ls")
]

for setting in tqdm(settings, desc="Test Linear Scaling"):
    ls, tuner, name = setting
    sys.stdout = open(f"log/log_scale_{name}.txt", 'w+')

    # Set functions and terminals
    functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
    terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node

    # Run GP
    sgp = SimpleGP(linear_scale=ls, tuner=tuner, functions=functions, pop_size=100, max_generations=100)

    CrossValidation(sgp, terminals).validate()
