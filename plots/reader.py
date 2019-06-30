import numpy as np


# Read average values of a log
def extract_mean_std(rest):
    split = rest.split(' ')
    return round(float(split[1]), 2), round(float(split[2]), 2)


def extract_value_per_run(log_name):
    with open(log_name, "r") as file_handle:
        line_list = file_handle.readlines()

    results = line_list[-4:len(line_list)]
    mean_train, std_train = extract_mean_std(results[1])
    mean_test, std_test = extract_mean_std(results[3])

    return mean_train, std_train / 9, mean_test, std_test


# Read all information from a log
def extract_all_per_gen(fname, runs=10, gens=100, pop_size=100):
    fitness = np.zeros((runs, gens))
    tree_size = np.zeros((runs, gens))
    before_fitness = np.zeros((runs, gens, pop_size))
    after_fitness = np.zeros((runs, gens, pop_size))

    run = None
    gen = 0
    tuner = 0

    with open(fname, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip("\n")

        # Parse the cross validation number
        if "Run" in line:
            run = int(line[-1])

        # Parse the generation information
        if "GA" in line:
            tuner = 0

            split = line.split(" ")

            fitness[run, gen] = split[2]
            tree_size[run, gen] = split[3]

            gen += 1
            if gen == gens:
                gen = 0

        # Parse tuner logs
        if "Tuner" in line and "Tuner converged" not in line:
            split = line.split(" ")

            before_fitness[run, gen, tuner] = split[1]
            after_fitness[run, gen, tuner] = split[2]

            tuner += 1

    # Average the tuner logs per gen
    # before_fitness[before_fitness == 0] = np.nan
    # # after_fitness[after_fitness == 0] = np.nan
    before_fitness = np.mean(before_fitness, axis=2)
    after_fitness = np.mean(after_fitness, axis=2)

    return fitness, tree_size, before_fitness, after_fitness


def extract_all_data(fnames, **kwargs):
    data = {}
    for idx, (fname, name) in enumerate(fnames):
        fit, size, after, before = extract_all_per_gen(fname, **kwargs)

        data[name] = (fit, size, after, before)

    return data


def extract_evaluation(file_name, runs=10):
    with open(file_name, "r") as file_handle:
        line_list = file_handle.readlines()

    evaluations = np.zeros(runs)

    for line in line_list:
        line = line.rstrip("\n")

        # Parse the cross validation number
        if "Run" in line:
            run = int(line[-1])

        if "evaluations" in line:
            split = line.split(' ')
            evaluations[run] = split[0]

    return evaluations


def avg_over_runs(run_data):
    return np.mean(run_data, axis=0), np.std(run_data, axis=0)
