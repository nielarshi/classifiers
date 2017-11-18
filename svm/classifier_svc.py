import numpy as np
from sklearn.svm import SVC
from data import sample_data_points_wo_bias, known_classifications
from plot_util import xy_plot, show_plot, get_ax


def train_model(sample_data, classifications):
    model = SVC(kernel='linear', C=1E10)
    model.fit(sample_data, classifications)

    return model


def plot_model_decision(model, ax, plot_support=True):
    x_limit = ax.get_xlim()
    y_limit = ax.get_ylim()

    new_x_array = np.linspace(x_limit[0], x_limit[1], 50)
    new_y_array = np.linspace(y_limit[0], y_limit[1], 50)

    y_mesh, x_mesh = np.meshgrid(new_y_array, new_x_array)
    xy = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T
    x_reshaped = model.decision_function(xy).reshape(x_mesh.shape)

    # plot decision boundary and margins
    ax.contour(x_mesh, y_mesh, x_reshaped, colors='k',
               levels=[-1, 0, 1], alpha=1,
               linestyles=['-'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')

ax = get_ax()
trained_model = train_model(sample_data_points_wo_bias, known_classifications)
xy_plot(sample_data_points_wo_bias, known_classifications)
plot_model_decision(trained_model, ax)
show_plot()
