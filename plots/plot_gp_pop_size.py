# sphinx_gallery_thumbnail_number = 3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from reader import extract_value_per_run

# Set theme
matplotlib.style.use('seaborn-darkgrid')
matplotlib.rcParams['font.family'] = "serif"

sizes = [10, 100, 500, 1000, 2000]

y_upper = [0] * len(sizes)
y = [0] * len(sizes)
y_lower = [0] * len(sizes)

z_upper = [0] * len(sizes)
z = [0] * len(sizes)
z_lower = [0] * len(sizes)

for idx, size in enumerate(sizes):
    mean_train, std_train, mean_test, std_test = extract_value_per_run(f"../log/log_pop_size_{size}.txt")
    y_upper[idx] = mean_train + std_train
    y[idx] = mean_train
    y_lower[idx] = mean_train - std_train

    z_upper[idx] = mean_test + std_test
    z[idx] = mean_test
    z_lower[idx] = mean_test - std_test

plt.figure()

plt.plot(np.log10(sizes), y, color='blue', label='MSE training')
plt.fill_between(np.log10(sizes), y_lower, y_upper, color='blue', alpha='0.1')

plt.xticks(np.log10(sizes), sizes)

plt.plot(np.log10(sizes), z, color='red', label='MSE test')
plt.fill_between(np.log10(sizes), z_lower, z_upper, color='red', alpha='0.1')

plt.xlabel('Population size')
plt.ylabel('MSE')

plt.title("Simple GP: MSE vs. population size")
plt.legend()
plt.savefig("../images/plot_gp_pop_size.png")
plt.show()
