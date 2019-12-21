import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import pvlib

# Constants
T = 51 + 273.15  # Temperature
Ns = 36  # Number of cells in series
Np = 1  # Number of cells in parallel
Vt = Ns * sc.Boltzmann * T / sc.elementary_charge  # Thermal voltage

# Data
currents = []
voltages = []  # TODO: ACQUIRE DATA


class Population:

    def __init__(self, size, genmax):
        self.gen = 1  # Generation number
        self.size = size
        self.genmax = genmax
        # Generate population from random vectors
        self.vectors = np.uniform
        self.average_fitness = None


class Vector:

    # Search space according to literature ([3] Oliva et al., 2017 and [4] Yu et al. 2017)
    a_L, a_H = 1, 2  # Ideality factor boundaries
    Rs_L, Rs_H = 0, 0.5  # Series resistance Rs

    def __init__(self, a, rs):
        # Search space parameters: (Ideality Factor, Series resistance)
        self.a = a
        self.rs = rs
        self.fitness = None

    def get_params(self, crit_points):
        """
        Re-extract the 5 parameters from each solution vector by formfitting on the pivot points
        :param crit_points: tuple of three critical points p1, p2 and p3 (MPP) respectively
        :return: tuple of the 5 single-diode parameters
        """
        p1, p2, p3 = crit_points
        alpha = p3[0] - p1[0] + (self.rs * (p3[1] - p1[1]) * (Ns / Np))
        beta = p2[0] - p1[0] + (self.rs * (p2[1] - p1[1]) * (Ns / Np))

        i0 = (alpha * (p2[1] - p1[1]) + beta * (p3[1] - p1[1])) / (Np * (alpha * (
                np.exp((p1[0] + p1[1] * self.rs * (Ns / Np)) / (self.a * Vt)) -
                np.exp((p2[0] + p2[1] * self.rs * (Ns / Np)) / (self.a * Vt))) + beta * (
                np.exp((p1[0] + p1[1] * self.rs * (Ns / Np)) / (self.a * Vt)) -
                np.exp((p3[0] + p3[1] * self.rs * (Ns / Np)) / (self.a * Vt)))))

        rp = ((p1[0] - p2[0]) * (Np / Ns) + self.rs * (p1[1] - p2[1])) / \
             (p2[1] - p1[1] - i0 * Np * (np.exp((p1[0] + p1[1] * self.rs * (Ns / Np)) / (self.a * Vt)) -
                                         np.exp((p2[0] + p2[1] * self.rs * (Ns / Np)) / (self.a * Vt))))

        ipv = (i0 * Np * (np.exp((p1[0] + p1[1] * self.rs * (Ns / Np)) / (self.a * Vt)) - 1) + (
                (p1[0] + p1[1] * self.rs * (Ns / Np)) / (rp * (Ns / Np))) + p1[1]) * (1 / Np)

        return rp, self.rs, self.a, i0, ipv

    def evaluate(self, crit_points):
        """
        Evaluates the fitness of the vector using the RMSE (J function)
        :param crit_points: tuple of three critical points p1, p2 and p3 (MPP) respectively
        :return: fitness of the vector
        """
        rp, self.rs, self.a, i0, ipv = self.get_params(crit_points)
        ical = pvlib.pvsystem.i_from_v(rp, self.rs, self.a * Vt, voltages, i0, ipv)
        error = np.abs(currents - ical)

        # Fitness penalty (J = 100) for unphysical values of Ipv, I0 and Rp
        if i0 < 0 or rp < 0 or ipv < 0:
            return 100

        return np.sqrt(np.mean(error ** 2))
