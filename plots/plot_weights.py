# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from reader import extract_all_data, avg_over_runs

# Set theme
matplotlib.style.use('seaborn-darkgrid')
matplotlib.rcParams['font.family'] = "serif"

# Global settings of the experiments
runs = 10
gens = 20
pop_size = 100
weights = [
    ("../log/log_weight_0_1.txt", "[0,1]"),
    ("../log/log_weight_0_5.txt", "[0,5]"),
    ("../log/log_weight_-5_5.txt", "[-5,5]"),
    # TODO: add the log below when finished
    # ("../log/log_weights_-1_1", "[-1,1]")
]

data = extract_all_data(weights, runs=runs, gens=gens, pop_size=pop_size)
x = np.arange(1, 21, 1)
# %%

palette = plt.get_cmap('tab10')
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

palette = plt.get_cmap('tab10')

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

palette = plt.get_cmap('tab20')

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

    plt.plot(x, m_after, color=palette(color + 1), label=f"After -{w}")
    plt.fill_between(x, m_after + std_after, m_after - std_after, color=palette(color + 1), alpha=0.1)

    # # TODO: this plot looks strange, it has a extreme outlier
    # delta = before - after
    # m_delta, std_delta = avg_over_runs(delta)
    #
    # plt.plot(x, m_delta, color=palette(color), label=w)
    # plt.fill_between(x, m_delta + std_delta, m_delta - std_delta, color=palette(color), alpha=0.1)

    color += 2

plt.legend()
plt.savefig("../images/weight_test_delta")
plt.show()
