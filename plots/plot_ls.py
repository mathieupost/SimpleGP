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
gens = 100
pop_size = 100
logs = [
    ("../log/log_scale_normal.txt", "Normal"),
    ("../log/log_scale_normal_ls.txt", "LS"),
]

data = extract_all_data(logs, runs=runs, gens=gens, pop_size=pop_size)
x = np.arange(1, gens + 1, 1)

#%%
# Plot best fitness per generation
palette = plt.get_cmap('tab10')

plt.figure()
plt.xlabel("Generation")
plt.ylabel("Best Training MSE")
# plt.xticks(x)
plt.title("Linear scaling: best fitness per generation")
plt.yscale("log")

color = 0
for w in data.keys():
    fit, _, _, _ = data[w]
    m_fit, std_fit = avg_over_runs(fit)
    plt.plot(x, m_fit, color=palette(color), label=w)
    plt.fill_between(x, m_fit + std_fit, m_fit - std_fit, color=palette(color), alpha=0.1)

    color += 1

plt.legend()
plt.savefig("../images/linear_scaling_fitness")
plt.show()

#%%

palette = plt.get_cmap('tab10')

# Plot size of the tree
plt.figure()
plt.xlabel("Generation")
plt.ylabel("Tree size")
plt.title("Linear scaling: average size of the trees per generation")

color = 0

for w in data.keys():
    _, size, _, _ = data[w]
    m_size, std_size = avg_over_runs(size)

    plt.plot(x, m_size, color=palette(color), label=w)
    plt.fill_between(x, m_size + std_size, m_size - std_size, color=palette(color), alpha=0.1)

    color += 1

plt.legend()
plt.savefig("../images/linear_scaling_size")
plt.show()


