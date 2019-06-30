# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from reader import extract_all_data, avg_over_runs, extract_evaluation, extract_value_per_run

# Set theme
matplotlib.style.use('seaborn-darkgrid')
matplotlib.rcParams['font.family'] = "serif"

# %%
# Global settings of the experiments
runs = 10
gens = 100
pop_size = 100
logs = [
    ("../log/log_scale_normal_100.txt", "Normal"),
    ("../log/log_scale_normal_ls_100.txt", "LS"),
    ("../log/log_tune_5_gen_max_gen_100.txt", "5 gen"),
    ("../log/log_tune_5_gen_ls_max_gen_100.txt", "5 gen - LS"),
    ("../log/log_tune_20_gen_max_gen_100.txt", "20 gen"),
    ("../log/log_tune_20_gen_ls_max_gen_100.txt", "20 gen - LS"),
    # ("../log/log_tune_20_gen_ls_max_gen_100.txt", "20 gen - LS - Lamarckian"),
    # ("../log/log_tune_20_gen_ls_max_gen_100_baldwin.txt", "20 gen - LS - Baldwin"),
]

data = extract_all_data(logs, runs=runs, gens=gens, pop_size=pop_size)
x = np.arange(1, gens + 1, 1)

# %%
# Plot best fitness per generation
palette = plt.get_cmap('tab10')

plt.figure()
plt.xlabel("Generation")
plt.ylabel("Best Training MSE")
plt.title("Tune after x generations: best fitness per generation")
# plt.yscale("log")

color = 0
for w in data.keys():
    fit, _, _, _ = data[w]
    m_fit, std_fit = avg_over_runs(fit)
    plt.plot(x, m_fit, color=palette(color), label=w)
    plt.fill_between(x, m_fit + std_fit, m_fit - std_fit, color=palette(color), alpha=0.1)

    color += 1

plt.legend()
plt.savefig("../images/tune_gen_fitness")
# plt.savefig("../images/tune_gen_fitness_baldwin")
plt.show()

# %%

palette = plt.get_cmap('tab10')

# Plot size of the tree
plt.figure()
plt.xlabel("Generation")
plt.ylabel("Tree size")
plt.title("Tune after x generations: tree size of elite per generation")

color = 0

for w in data.keys():
    _, size, _, _ = data[w]
    m_size, std_size = avg_over_runs(size)

    plt.plot(x, m_size, color=palette(color), label=w)
    plt.fill_between(x, m_size + std_size, m_size - std_size, color=palette(color), alpha=0.1)

    color += 1

plt.legend()
plt.savefig("../images/tune_gen_size")
# plt.savefig("../images/tune_gen_size_baldwin")
plt.show()

# %%

# Plots for 5gens tuning and 20gens tuning can better be seperated...

x_gens5 = list(range(4, 100, 5))
x_gens20 = list(range(19, 100, 20))

palette = plt.get_cmap('tab20')
color = 0


def get_before(key):
    _, _, before, _ = data[key]
    m_before, std_before = avg_over_runs(before)

    y = m_before[m_before > 0]
    std = std_before[std_before > 0]

    return y, std


def get_after(key):
    _, _, _, after = data[key]
    m_after, std_after = avg_over_runs(after)

    y = m_after[m_after > 0]
    std = std_after[std_after > 0]

    return y, std


# %% Plot average fitness before

def plot_before(keys, tune_gens, name):
    plt.figure()
    plt.xlabel("Generation")
    plt.ylabel("Training MSE")
    plt.title("Tune after x generations: MSE of the population before tuning")
    plt.yscale("log")

    palette = plt.get_cmap('tab10')

    color = 0

    for k, gens in zip(keys, tune_gens):
        y_before, std_before = get_before(k)

        x = np.arange(len(gens))

        plt.plot(x, y_before, color=palette(color), label=f"{k}")
        plt.fill_between(x, y_before + std_before, y_before - std_before, color=palette(color), alpha=0.1)

        # Increment due to 0-index
        plt.xticks(x, list(map(lambda x: x + 1, gens)))

        color += 1

    # Log scale so log(1) = 0
    plt.ylim([1, plt.ylim()[1]])
    plt.legend()
    plt.savefig(f"../images/{name}")
    plt.show()


## Seperate plots of 20 gens and 5 gens
plot_before(['5 gen', '5 gen - LS'], list([x_gens5, x_gens5]), "tune_5_gen_before")
plot_before(['20 gen', '20 gen - LS'], list([x_gens20, x_gens20]), "tune_20_gen_before")


# %% Plot after fitness after tuning


def plot_after(keys, tune_gens, name):
    plt.figure()
    plt.xlabel("Generation")
    plt.ylabel("Training MSE")
    plt.title("Tune after x generations: MSE of the population after tuning")

    palette = plt.get_cmap('tab10')

    color = 0

    for k, gens in zip(keys, tune_gens):
        y_after, std_after = get_after(k)

        x = np.arange(len(gens))

        plt.plot(x, y_after, color=palette(color), label=f"{k}")
        plt.fill_between(x, y_after + std_after, y_after - std_after, color=palette(color), alpha=0.1)

        # Increment due to 0-index
        plt.xticks(x, list(map(lambda x: x + 1, gens)))

        color += 1

    plt.ylim([0, plt.ylim()[1]])
    plt.legend()
    plt.savefig(f"../images/{name}")
    plt.show()


## Seperate plots of 20 gens and 5 gens
plot_after(['5 gen', '5 gen - LS'], list([x_gens5, x_gens5]), "tune_5_gen_after")
plot_after(['20 gen', '20 gen - LS'], list([x_gens20, x_gens20]), "tune_20_gen_after")


# %%
def bar_evaluations():
    palette = plt.get_cmap('tab10')
    plt.figure()
    plt.xlabel("Settings")
    plt.ylabel("Evaluations")
    plt.yscale("log")
    plt.title("Tune after x generations: Evaluations")
    labels = []
    means = []
    stds = []
    for l in logs:
        file, label = l
        labels.append(label)
        evals = extract_evaluation(file)
        m_eval, std_eval = avg_over_runs(evals)
        means.append(m_eval)
        stds.append(std_eval)
    bars = plt.bar(labels, means, yerr=stds, capsize=10, bottom=10 ** 4, color=[palette(i) for i in range(6)],
                   alpha=0.7)
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, (height + 10 ** 4) * 0.65, human_format(height), ha='center',
                 va='center', color='white', weight='bold')

    plt.savefig("../images/tune_evaluations")
    plt.show()


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


bar_evaluations()


# %%
def bar_mse():
    palette = plt.get_cmap('tab10')
    fig, ax = plt.subplots()
    ax.set_xlabel("Settings")
    ax.set_ylabel("MSE")
    ax.set_title("Tune after x generations: MSE")
    labels = []
    means_train = []
    stds_train = []
    means_test = []
    stds_test = []
    for l in logs:
        file, label = l
        labels.append(label)
        mean_train, std_train, mean_test, std_test = extract_value_per_run(file)
        means_train.append(mean_train)
        stds_train.append(std_train)
        means_test.append(mean_test)
        stds_test.append(std_test)

    width = 0.35
    x = np.array(range(len(labels)))
    train_bars = ax.bar(x - width / 2, means_train, width, yerr=stds_train, capsize=10,
                        color=[palette(i) for i in range(6)], alpha=0.4)
    test_bars = ax.bar(x + width / 2, means_test, width, yerr=stds_test, capsize=10,
                       color=[palette(i) for i in range(6)],
                       alpha=0.7)

    for label, rect in zip(["train"] * len(x) + ["test"] * len(x), train_bars + test_bars):
        plt.text(rect.get_x() + rect.get_width() / 2.0, 2, label, ha='center',
                 va='bottom', color='white', weight='bold', rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    fig.savefig("../images/tune_mse")
    fig.show()


bar_mse()
