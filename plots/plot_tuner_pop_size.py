# %%
import numpy as np

from reader import avg_over_runs, extract_all_data

runs = 10
pop_size = 100


def extract_numbers(file_name):
    with open(file_name, "r") as file_handle:
        line_list = file_handle.readlines()

    mse = np.zeros((runs, pop_size))
    iterations = np.zeros((runs, pop_size))
    # TODO: analyse the time it took
    evaluations = np.zeros(runs)

    run = None
    pop = None

    for line in line_list:
        line = line.rstrip("\n")

        # Parse the cross validation number
        if "Run" in line:
            run = int(line[-1])
            pop = 0
            stopped_early = False

        # Parse tuner logs
        if "Tuner" in line:
            if "converged" in line:
                split = line.split(' ')
                iterations[run, pop] = split[-1]
                stopped_early = True
            else:
                split = line.split(' ')

                if not stopped_early:
                    # Max iteration was reached thus 100
                    iterations[run, pop] = 100

                mse[run, pop] = float(split[2])
                stopped_early = False
                pop += 1

        if "evaluations" in line:
            split = line.split(' ')
            evaluations[run] = split[0]

    # Average over the population
    mse = np.mean(mse, axis=1)
    iterations = np.mean(iterations, axis=1)

    return mse, iterations, evaluations


data = {}
files = []
sizes = [10, 20, 50, 100, 500]
# sizes = [50, 100]

for size in sizes:
    file_name = f"../log/log_tuner_pop_size_{size}.txt"
    files.append((file_name, f"tuner_pop_{size}"))
    data[size] = extract_numbers(file_name)

# %%

import matplotlib.pyplot as plt

# Set theme
import matplotlib

matplotlib.style.use('seaborn-darkgrid')
matplotlib.rcParams['font.family'] = "serif"

# Create y values
ys_evals = np.zeros_like(sizes)
ys_evals_std = np.zeros_like(sizes)
for idx, s in enumerate(sizes):
    _, _, evals = data[s]
    m_eval, std_eval = avg_over_runs(evals)
    ys_evals[idx] = m_eval
    ys_evals_std[idx] = std_eval

# %%
ys_mse_before = []
ys_mse_after = []
ys_std_before = []
ys_std_after = []
all_data = extract_all_data(files, runs=runs, gens=100, pop_size=pop_size)
for w in all_data.keys():
    _, _, before, after = all_data[w]
    ys_mse_before.append(before[:, -1].mean())
    ys_std_before.append(before[:, -1].std())
    ys_mse_after.append(after[:, -1].mean())
    ys_std_after.append(after[:, -1].std())
ys_mse_before = np.array(ys_mse_before)
ys_mse_after = np.array(ys_mse_after)
ys_std_before = np.array(ys_std_before)
ys_std_after = np.array(ys_std_after)
# %%

palette = plt.get_cmap('tab10')
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xticks(sizes)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.plot(sizes, ys_mse_before, label="MSE before tuning", color=palette(0))
ax.fill_between(sizes, ys_mse_before + ys_std_before, ys_mse_before - ys_std_before, color=palette(0), alpha=0.1)
ax.plot(sizes, ys_mse_after, label="MSE after tuning", color=palette(1))
ax.fill_between(sizes, ys_mse_after + ys_std_after, ys_mse_after - ys_std_after, color=palette(1), alpha=0.1)

ax.set_xlabel("Tuner population size")
ax.set_ylabel("MSE")
ax.set_title("Tuner: MSE vs population size")
ax.legend()

fig.savefig("../images/tuner_pop_size_mse")
fig.show()

#%%
plt.figure()
plt.plot(sizes, ys_evals, label="Total evaluations")
plt.fill_between(sizes, ys_evals + ys_evals_std, ys_evals - ys_evals_std, alpha=0.1)

plt.xlabel("Tuner population size")
plt.ylabel("Evaluations")
plt.title("Tuner: evaluations vs population size")
plt.legend()

plt.savefig("../images/tuner_pop_size_evals")
plt.show()
