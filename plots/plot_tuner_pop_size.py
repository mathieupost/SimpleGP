# %%
import numpy as np

from reader import avg_over_runs

runs = 10
pop_size = 100


def extract_numbers(size):
    with open(f"../log/log_tuner_pop_size_{size}.txt", "r") as file_handle:
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

                mse[run, pop] = float(split[-1])
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
sizes = [50, 100, 500]
# sizes = [50, 100]

for size in sizes:
    data[size] = extract_numbers(size)

# %%

import matplotlib.pyplot as plt

# Set theme
import matplotlib

matplotlib.style.use('seaborn-darkgrid')
matplotlib.rcParams['font.family'] = "serif"

# Create y values
ys_mse = np.zeros_like(sizes)
ys_evals = np.zeros_like(sizes)
for idx, s in enumerate(sizes):
    mse, _, evals = data[s]
    m_mse, _ = avg_over_runs(mse)
    m_eval, _ = avg_over_runs(evals)

    ys_mse[idx] = m_mse
    ys_evals[idx] = m_eval

plt.figure()
plt.plot(sizes, ys_mse, color='blue', label="MSE after tuning")

plt.xlabel("Tuner population size")
plt.ylabel("MSE")
plt.title("Tuner: MSE vs population size")
plt.legend()

print(ys_mse)
print(ys_evals)

plt.savefig("../images/tuner_pop_size_mse_50_100_500")
plt.show()

plt.figure()
plt.plot(sizes, ys_evals, color='blue', label="Total evaluations")

plt.xlabel("Tuner population size")
plt.ylabel("Evaluations")
plt.title("Tuner: evaluations vs population size")
plt.legend()

plt.savefig("../images/tuner_pop_size_evals_50_100_500")
plt.show()
