import matplotlib.pyplot as plt


# plot using matplotlib's pyplot
def xy_plot(sample_data, classification):
    for index, sample_data_unit in enumerate(sample_data):
        # Plot the negative samples (x array's 3rd entry is checked for -1)
        if classification[index] == -1:
            plt.plot(sample_data_unit[0], sample_data_unit[1], "_")
        # Plot the positive samples (x array's 3rd entry is checked for +1)
        elif classification[index] == 1:
            plt.plot(sample_data_unit[0], sample_data_unit[1], "+")


def show_plot():
    plt.show()


def get_ax():
    return plt.gca()
