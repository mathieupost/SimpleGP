# sphinx_gallery_thumbnail_number = 3
import matplotlib.pyplot as plt
import numpy as np


def extract_mean_std(rest):
    split = rest.split(' ')
    return round(float(split[1]), 2), round(float(split[2]), 2)


def extract_value_per_run(size):
    with  open(f"../log/log_pop_size_{size}.txt", "r") as file_handle:
        line_list = file_handle.readlines()

    results = line_list[-4]
    mean_train, std_train = extract_mean_std(results[1])
    mean_test, std_test = extract_mean_std(results[3])

    return mean_train, std_train / 9, mean_test, std_test


sizes = [10, 100, 500, 1000, 2000]

y_upper = [0] * len(sizes)
y = [0] * len(sizes)
y_lower = [0] * len(sizes)

z_upper = [0] * len(sizes)
z = [0] * len(sizes)
z_lower = [0] * len(sizes)

for idx, size in enumerate(sizes):
    mean_train, std_train, mean_test, std_test = extract_value_per_run(size)
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
plt.savefig("../images/plot_gp_pop_size.png")
plt.show()
