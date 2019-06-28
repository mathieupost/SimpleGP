# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Set theme
matplotlib.style.use('seaborn-darkgrid')
matplotlib.rcParams['font.family'] = "serif"

# Global settings of the experiments
runs = 10
gens = 20
pop_size = 100
weights = [(0, 1, "[0,1]"), (-5, 5, "[-5,5]")]


def extract_avg_mean_std(rest):
    split = rest.split(' ')
    return round(float(split[1]), 2), round(float(split[2]), 2)


def extract_fitness_per_gen(lower, upper):
    fitness = np.zeros((runs, gens))
    tree_size = np.zeros((runs, gens))
    before_fitness = np.zeros((runs, gens, pop_size))
    after_fitness = np.zeros((runs, gens, pop_size))

    run = None
    gen = 0
    tuner = 0

    with open(f"../log/log_weight_{lower}_{upper}.txt", "r") as f:
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
            gen = int(split[1]) - 1

            fitness[run, gen] = split[2]
            tree_size[run, gen] = split[3]

        # Parse tuner logs
        if "Tuner" in line and "Tuner converged" not in line:
            split = line.split(" ")

            before_fitness[run, gen, tuner] = split[1]
            after_fitness[run, gen, tuner] = split[2]

            tuner += 1

    # Average the tuner logs per gen
    before_fitness = np.mean(before_fitness, axis=2)
    after_fitness = np.mean(after_fitness, axis=2)

    return fitness, tree_size, before_fitness, after_fitness


def extract_all_data(weights):
    data = {}
    for idx, (lower, upper, name) in enumerate(weights):
        fit, size, after, before = extract_fitness_per_gen(lower, upper)

        data[name] = (fit, size, after, before)

    return data


def avg_over_runs(run_data):
    return np.mean(run_data, axis=0), np.std(run_data, axis=0)


data = extract_all_data(weights)
x = np.arange(1, 21, 1)
palette = plt.get_cmap('Set1')
# %%

# Plot best fitness per generation

plt.figure()
plt.xlabel("Generation")
plt.ylabel("Best Training MSE")
plt.xticks(x)
plt.title("Weight Tuning: best fitness per generation")

color = 0
for w in data.keys():
    fit, _, _, _ = data[w]
    m_fit, std_fit = avg_over_runs(fit)
    plt.plot(x, m_fit, color=palette(color), label=w)
    plt.fill_between(x, m_fit + std_fit, m_fit - std_fit, color=palette(color), alpha=0.1)

    color += 1

plt.legend()
plt.savefig("../images/weight_test_fitness")
plt.show()

# %%

# Plot size of the tree
plt.figure()
plt.xlabel("Generation")
plt.ylabel("Tree size")
plt.xticks(x)
plt.title("Weight Tuning: average size of the trees per generation")

color = 0

for w in data.keys():
    _, size, _, _ = data[w]
    m_size, std_size = avg_over_runs(size)

    plt.plot(x, m_size, color=palette(color), label=w)
    plt.fill_between(x, m_size + std_size, m_size - std_size, color=palette(color), alpha=0.1)

    color += 1

plt.legend()
plt.savefig("../images/weight_test_size")
plt.show()

# %%

# Plot the average improvement with weight tuning
plt.figure()
plt.xlabel("Generation")
plt.ylabel("Training MSE")
plt.xticks(x)
plt.title("Weight Tuning: ")

plt.ylim([0, 1500])
color = 0

# If the plots are combined, you wont see the difference for the smaller one

# select = list(data.keys())[1]
# for w in [select]:
for w in data.keys():
    _, _, before, after = data[w]

    m_before, std_before = avg_over_runs(before)
    m_after, std_after = avg_over_runs(after)

    plt.plot(x, m_before, color=palette(color), label=f"Before -{w}")
    # plt.fill_between(x, m_before + std_before, m_before - std_before, color=palette(color), alpha=0.1)

    plt.plot(x, m_after, color=palette(color + 2), label=f"After -{w}")
    plt.fill_between(x, m_after + std_after, m_after - std_after, color=palette(color + 2), alpha=0.1)

    # # TODO: this plot looks strange, it has a extreme outlier
    # delta = before - after
    # m_delta, std_delta = avg_over_runs(delta)
    #
    # plt.plot(x, m_delta, color=palette(color), label=w)
    # plt.fill_between(x, m_delta + std_delta, m_delta - std_delta, color=palette(color), alpha=0.1)

    color += 1

plt.legend()
plt.savefig("../images/weight_test_delta")
plt.show()
