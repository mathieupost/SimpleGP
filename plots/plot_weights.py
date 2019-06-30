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
    ("../log/log_weight_-1_1.txt", "[-1,1]"),
    ("../log/log_weight_0_5.txt", "[0,5]"),
    ("../log/log_weight_-5_5.txt", "[-5,5]"),
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

palette = plt.get_cmap('tab10')

# Plot the average fitness before weight tuning
plt.figure()
plt.xlabel("Generation")
plt.ylabel("Training MSE")
plt.xticks(x)
plt.title("Weight Tuning: average fitness before")
plt.yscale("log")

color = 0

for w in data.keys():
    _, _, before, _ = data[w]

    m_before, std_before = avg_over_runs(before)
    plt.plot(x, m_before, color=palette(color), label=w)
    plt.fill_between(x, m_before + std_before, m_before - std_before, color=palette(color), alpha=0.1)

    color += 1

plt.legend()
plt.savefig("../images/weight_test_before")
plt.show()

# %%

palette = plt.get_cmap('tab10')

# Plot the average fitness after weight tuning
plt.figure()
plt.xlabel("Generation")
plt.ylabel("Training MSE")
plt.xticks(x)
plt.title("Weight Tuning: average fitness after")
plt.yscale("log")

color = 0

for w in data.keys():
    _, _, _, after = data[w]
    m_after, std_after = avg_over_runs(after)

    plt.plot(x, m_after, color=palette(color), label=w)
    plt.fill_between(x, m_after + std_after, m_after - std_after, color=palette(color), alpha=0.1)

    color += 1

plt.legend()
plt.savefig("../images/weight_test_after")
plt.show()
