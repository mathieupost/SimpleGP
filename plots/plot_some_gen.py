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
logs = [
    # TODO: add data from log below when finished
    # ("../log/log_tune_5_gen.txt", "5 gen"),
    ("../log/log_tune_5_gen_ls.txt", "5 gen - LS"),
    ("../log/log_tune_20_gen.txt", "20 gen"),
    ("../log/log_tune_20_gen_ls.txt", "20 gen - LS"),
]

data = extract_all_data(logs, runs=runs, gens=gens, pop_size=pop_size)
x = np.arange(1, gens + 1, 1)

#%%
# Plot best fitness per generation
palette = plt.get_cmap('tab10')

plt.figure()
plt.xlabel("Generation")
plt.ylabel("Best Training MSE")
plt.xticks(x)
plt.title("Tune after x generations: best fitness per generation")

color = 0
for w in data.keys():
    fit, _, _, _ = data[w]
    m_fit, std_fit = avg_over_runs(fit)
    plt.plot(x, m_fit, color=palette(color), label=w)
    plt.fill_between(x, m_fit + std_fit, m_fit - std_fit, color=palette(color), alpha=0.1)

    color += 1

plt.legend()
plt.savefig("../images/tune_gen_fitness")
plt.show()

#%%

palette = plt.get_cmap('tab10')

# Plot size of the tree
plt.figure()
plt.xlabel("Generation")
plt.ylabel("Tree size")
plt.title("Tune after x generations: average size of the trees per generation")

color = 0

for w in data.keys():
    _, size, _, _ = data[w]
    m_size, std_size = avg_over_runs(size)

    plt.plot(x, m_size, color=palette(color), label=w)
    plt.fill_between(x, m_size + std_size, m_size - std_size, color=palette(color), alpha=0.1)

    color += 1

plt.legend()
plt.savefig("../images/tune_gen_size")
plt.show()

# %%

# Plots for 5gens tuning and 20gens tuning can better be seperated...

gens5 = ['5 gen - LS']
x_gens5 = [4, 9, 14, 19]
gens20 = ['20 gen', '20 gen - LS']
x_gens20 = [19]

palette = plt.get_cmap('tab20')
color = 0


def plot_gen_diff(x, keys, name, color):
    # Plot the average improvement with weight tuning
    plt.figure()
    plt.xlabel("Generation")
    plt.ylabel("Training MSE")
    plt.xticks(x)
    plt.title("Weight Tuning: fitness before and after tuning")

    # plt.xlim([1, 20])
    # plt.ylim([0, 1500])

    for w in keys:
        _, _, before, after = data[w]

        # Mask zeros values
        before = before[before > 0]
        after = after[after > 0]
        print(before.shape)

        m_before, std_before = avg_over_runs(before)

        m_after, std_after = avg_over_runs(after)

        plt.plot(x, m_before, color=palette(color), label=f"Before -{w}")
        # plt.fill_between(x, m_before + std_before, m_before - std_before, color=palette(color), alpha=0.1)

        plt.plot(x, m_after, color=palette(color + 1), label=f"After -{w}")
        # plt.fill_between(x, m_after + std_after, m_after - std_after, color=palette(color + 1), alpha=0.1)

        color += 2

    plt.legend()
    plt.savefig(f"../images/tune_gen_{name}")
    plt.show()

plot_gen_diff(x_gens5, gens5, "gens5", 0)
plot_gen_diff(x_gens20, gens20, "gens20", 2)

