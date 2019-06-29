# sphinx_gallery_thumbnail_number = 3
import matplotlib.pyplot as plt
import numpy as np


def extract_numbers(size):
    with open(f"../log/log_tuner_pop_size_{size}.txt", "r") as file_handle:
        line_list = file_handle.readlines()

    stopped_at = 0
    mse_sum = 0
    mse_items = 0
    for line in line_list:
        if "Tuner" in line:
            if "converged" in line:
                stopped_at += int(line.split(' ')[-1])
            else:
                mse_sum += (float(line.split(' ')[-2]) - float(line.split(' ')[-1]))
                mse_items += 1
                stopped_at += 100

    print(stopped_at / mse_items)
    print(mse_sum / mse_items)
    print()


sizes = [10, 20, 50, 100]

for size in sizes:
    extract_numbers(size)

# plt.figure()
#
# plt.plot(np.log10(sizes), y, color='blue', label='MSE training')
# plt.fill_between(np.log10(sizes), y_lower, y_upper, color='blue', alpha='0.1')
#
# plt.xticks(np.log10(sizes), sizes)
#
# plt.plot(np.log10(sizes), z, color='red', label='MSE test')
# plt.fill_between(np.log10(sizes), z_lower, z_upper, color='red', alpha='0.1')
#
# plt.xlabel('Population size')
# plt.ylabel('MSE')
#
# plt.title("Simple GP: MSE vs. population size")
# plt.savefig("../images/plot_gp_pop_size.png")
# plt.show()
