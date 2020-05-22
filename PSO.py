import numpy as np
import pyswarms as ps
from objects import DE, read_csv
from scipy.constants import constants
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt


def obj_func(x, exp_data, vth, Ns, Np):
    """
    :param x:
    :return:
    """
    if not x.shape[1] == 5:
        raise IndexError("obj function only takes 5-dimensional input not {}.".format(x.shape[1]))

    j = np.asarray([DE.objf(particle, exp_data, vth, Ns, Np) for particle in x])

    return j


temp = 33 + 275.15

exp_data = read_csv("data/RTC33D1000W.csv")
vth = constants.Boltzmann * temp / constants.elementary_charge
Ns, Np = 1, 1

kwargs = {'exp_data': exp_data, 'vth': vth, 'Ns': Ns, 'Np': Np}

# Bounds
xmax = np.array([60, 0.1, 1.5, 4e-07, 0.8])
xmin = np.array([50, 0, 1.4, 3e-07, 0.7])
bounds = (xmin, xmax)

options = {'c1': 2, 'c2': 2, 'w': 0.4}

optimizer = ps.single.GlobalBestPSO(n_particles=200, dimensions=5, options=options, bounds=bounds)

cost, pos = optimizer.optimize(obj_func, iters=500, **kwargs)

plot_cost_history(cost_history=optimizer.cost_history)
plt.show()


# animation
# def wrap_func(x):
#     return obj_func(x, exp_data, vth, Ns, Np)
#
#
# m = Mesher(func=wrap_func)
#
# animation = plot_contour(pos_history=optimizer.pos_history,
#                          mesher=m,
#                          mark=(0, 0))
#
# animation.save('plot0.gif', writer='imagemagick', fps=10)
# Image(url='plot0.gif')