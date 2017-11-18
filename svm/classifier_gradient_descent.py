import numpy as np
from data import sample_data_points_with_bias, known_classifications
from plot_util import xy_plot, get_ax, show_plot


def svm_gd(sample_data, classification):
    weights = np.zeros(len(sample_data[0]))
    learning_rate = 1
    iteration_count = 100000  # epochs
    errors = []

    for iteration_num in range(1, iteration_count):  # 1 to avoid divide by 0
        error = 0
        for index, sample_data_unit in enumerate(sample_data):
            '''
                Points are in three categories:
                1. y*f(x) > 1
                    Point is outside margin.
                    No contribution to loss
                2. y*f(x)=1
                    Point is on margin.
                    No contribution to loss.
                    As in hard margin case.
                3. y*f(x) < 1
                    Point violates margin constraint.
                    Contributes to loss
            '''
            if (classification[index] * np.dot(sample_data_unit, weights)) < 1:
                weights = weights + learning_rate * \
                                    ((sample_data_unit * classification[index]) +
                                     (-2 * (1/iteration_num) * weights))
                error = 1
            else:
                weights = weights + learning_rate * (-2 * (1/iteration_num) * weights)

        errors.append(error)
    return weights


def train_and_plot(sample_data, classification, ax):
    calculated_weights = svm_gd(sample_data, classification)
    # Print the hyperplane calculated by svm_gd()
    x2 = [calculated_weights[0], calculated_weights[1], -calculated_weights[1], calculated_weights[0]]
    x3 = [calculated_weights[0], calculated_weights[1], calculated_weights[1], -calculated_weights[0]]
    x2x3 = np.array([x2, x3])
    x, y, u, v = zip(*x2x3)
    ax.quiver(x, y, u, v, scale=1, color='blue')


ax = get_ax()
xy_plot(sample_data_points_with_bias, known_classifications)
train_and_plot(sample_data_points_with_bias, known_classifications, ax)
show_plot()
