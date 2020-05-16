import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
from scipy.special import lambertw


# Reads experimental IV data from a csv file
def read_csv(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)  # Ignore the header
        data = [(float(row[0]), float(row[1])) for row in csv_reader]
        voltages, currents = np.array([row[0] for row in data]), np.array([row[1] for row in data])
        return voltages, currents


class DE:

    def __init__(self, bounds, ivdata, Ns, Np, popsize=100, maxiter=200, mutf=0.7, crossr=0.8):
        """
        :param bounds: Dictionary of bounds in this form {'rp': [lower, upper], 'rs': [lower, upper] ...}
        :param ivdata: [Voltages, Currents]
        """
        self.bounds = bounds  #
        self.popsize = popsize
        self.maxiter = maxiter
        self.mutf = mutf
        self.crossr = crossr
        self.dim = len(bounds)  # Dimensions of the search space
        self.populations = self.maxiter * [self.popsize * [None]]  # Initial populations
        self.fitnesses = self.maxiter * [self.popsize * [100]]  # Initial fitnesses
        self.ivdata = ivdata
        self.Ns = Ns
        self.Np = Np
        self.final_res = (None, None)  # Final result containing (vector, fitness)

    # Main differential evolution algorithm
    def solve(self, temp):
        vth = constants.Boltzmann * temp / constants.elementary_charge
        # Initial uniform distribution
        self.populations[0] = np.random.uniform([i[0] for i in self.bounds.values()],
                                                [j[1] for j in self.bounds.values()], size=(self.popsize, self.dim))
        for gen in range(self.maxiter):
            # Initial fitness
            self.fitnesses[gen] = [DE.objf(vec, self.ivdata, vth, self.Ns, self.Np) for vec in self.populations[gen]]

            fittest_index = np.argmin(self.fitnesses[gen])
            fittest = self.populations[gen][fittest_index]

            # Iterate over population
            for i in range(self.popsize):
                # Generate donor vector from 2 random vectors and the fittest
                indices = [ind for ind in range(self.popsize) if ind != i and ind != fittest_index]
                i1, i2 = np.random.choice(indices, 2, replace=False)

                # DE mutation
                mutant = fittest + self.mutf * (self.populations[gen][i1] - self.populations[gen][i2])

                # Crossover
                crosspoints = np.random.rand(self.dim) <= self.crossr
                if not any(crosspoints):
                    crosspoints[np.random.randint(0, self.dim)] = True
                trial = np.where(crosspoints, mutant, self.populations[gen][i])

                # Penalty
                for j, key in enumerate(self.bounds.keys()):
                    lower, upper = self.bounds[key]
                    # Solution vectors need to be [rp, rs, a, i0, iph]
                    if not lower <= trial[j]:
                        trial[j] = lower + np.random.rand() * (upper - lower)
                    if not upper >= trial[j]:
                        trial[j] = upper - np.random.rand() * (upper - lower)

                # Selection
                f = self.objf(trial, self.ivdata, vth, self.Ns, self.Np)

                if gen + 1 < self.maxiter:
                    if f < self.fitnesses[gen][i]:
                        #  Select for next generation
                        self.populations[gen + 1][i] = trial
                        self.fitnesses[gen + 1][i] = f
                    else:
                        self.populations[gen + 1][i] = self.populations[gen][i]

    @staticmethod
    def objf(vector, exp_data, vth, Ns, Np):
        # If vector is none, max fitness
        if vector is None:
            return 100
        voltages = exp_data[0]
        ical = DE.i_from_v(vector, voltages, vth, Ns, Np)
        delta = ical - exp_data[1]  # compare currents for each voltage value
        # Fitness penalty (J = 100) for unphysical values of Rs and Rp
        if vector[0] < 0 or vector[1] < 0:
            return 100
        return np.sqrt(np.mean(delta ** 2))

    @staticmethod
    def i_from_v(vector, v, vth, Ns, Np):
        # Takes a solution vector of 5/7 params and a voltage to calculate the current
        if vector is not None:
            if len(vector) != 5 and len(vector) != 7:
                print("V:{}".format(vector))
                print("Vector has {} parameters! needs 5 or 7.\n Exiting..".format(len(vector)))
                exit(0)

        output_is_scalar = np.isscalar(v)

        # For numerical stability: Gsh=1/Rsh, so Rsh=np.inf ==> Gsh=0
        conductance_shunt = 1. / (vector[0] * (Ns / Np))
        #  Single diode
        if len(vector) == 5:
            resistance_shunt, resistance_series, n, saturation_current, photocurrent = vector

            Gsh, Rs, a, V, I0, IL = np.broadcast_arrays(conductance_shunt, resistance_series * (Ns / Np),
                                                        n * Ns * vth, v, saturation_current * Np,
                                                        photocurrent * Np)

            # Initialize output I (V might not be float64)
            current = np.full_like(V, np.nan, dtype=np.float64)  # noqa: E741, N806

            # Determine indices where 0 < Rs requires implicit model solution
            idx_p = 0. < Rs

            # Determine indices where 0 = Rs allows explicit model solution
            idx_z = 0. == Rs

            # Explicit solutions where Rs=0
            if np.any(idx_z):
                current[idx_z] = IL[idx_z] - I0[idx_z] * np.expm1(V[idx_z] / a[idx_z]) - \
                           Gsh[idx_z] * V[idx_z]

            # possibility of overflow
            if np.any(idx_p):
                # LambertW argument
                argW = Rs[idx_p] * I0[idx_p] / (a[idx_p] * (Rs[idx_p] * Gsh[idx_p] + 1.)) * \
                       np.exp((Rs[idx_p] * (IL[idx_p] + I0[idx_p]) + V[idx_p]) /
                              (a[idx_p] * (Rs[idx_p] * Gsh[idx_p] + 1.)))

                # lambertw typically returns complex value with zero imaginary part
                # may overflow to np.inf
                lambertwterm = lambertw(argW).real

                # Jain and Kapoor, 2004
                current[idx_p] = (IL[idx_p] + I0[idx_p] - V[idx_p] * Gsh[idx_p]) / \
                                 (Rs[idx_p] * Gsh[idx_p] + 1.) - (a[idx_p] / Rs[idx_p]) * lambertwterm

            if output_is_scalar:
                return current.item()
            else:
                return current

        # Double diode
        if len(vector) == 7:
            resistance_shunt, resistance_series, n1, n2, saturation_current1, saturation_current2, photocurrent = vector
            Gsh, Rs, a1, a2, V, I01, I02, IL = np.broadcast_arrays(conductance_shunt, resistance_series, n1 * Ns * vth,
                                                        n2 * Ns * vth, v, saturation_current1 * Np,
                                                        saturation_current2 * Np, photocurrent * Np)
            # Intitalize output I (V might not be float64)
            current = np.full_like(V, np.nan, dtype=np.float64)
            # Determine indices where 0 < Rs requires implicit model solution
            idx_p = 0. < Rs

            # Determine indices where 0 = Rs allows explicit model solution
            idx_z = 0. == Rs

            # Explicit solutions where Rs=0
            if np.any(idx_z):
                current[idx_z] = IL[idx_z] - I01[idx_z] * np.expm1(V[idx_z] / a1[idx_z]) - \
                                 I02[idx_z] * np.expm1(V[idx_z] / a2[idx_z]) - \
                                 Gsh[idx_z] * V[idx_z]

            # Only compute using LambertW if there are cases with Rs>0
            # Does NOT handle possibility of overflow, github issue 298
            if np.any(idx_p):
                # LambertW arguments, cannot be float128, may overflow to np.inf
                argw1 = (Rs[idx_p] * (I01[idx_p] + I02[idx_p]) / (a1[idx_p] * (Gsh[idx_p] * Rs[idx_p] + 1))) * \
                        np.exp((Rs[idx_p] * (IL[idx_p] + I01[idx_p] + I02[idx_p]) + V[idx_p]) /
                               (a1[idx_p] * (Gsh[idx_p] * Rs[idx_p] + 1))
                )
                argw2 = (Rs[idx_p] * (I01[idx_p] + I02[idx_p]) / (a2[idx_p] * (Gsh[idx_p] * Rs[idx_p] + 1))) * \
                        np.exp((Rs[idx_p] * (IL[idx_p] + I01[idx_p] + I02[idx_p]) + V[idx_p]) /
                               (a2[idx_p] * (Gsh[idx_p] * Rs[idx_p] + 1))
                )
                # lambertw typically returns complex value with zero imaginary part
                # may overflow to np.inf
                lambertwterm1 = lambertw(argw1).real
                lambertwterm2 = lambertw(argw2).real

                # Eqn. 2 in Jain and Kapoor, 2004
                #  I = -V/(Rs + Rsh) - (a/Rs)*lambertwterm + Rsh*(IL + I0)/(Rs + Rsh)
                # Recast in terms of Gsh=1/Rsh for better numerical stability.
                # Eqn. 2 in Jain and Kapoor, 2004
                #  I = -V/(Rs + Rsh) - (a/Rs)*lambertwterm + Rsh*(IL + I0)/(Rs + Rsh)
                # Recast in terms of Gsh=1/Rsh for better numerical stability.
                current[idx_p] = (IL[idx_p] + I01[idx_p] + I02[idx_p] - (V[idx_p] * Gsh[idx_p])) / \
                                 (Gsh[idx_p] * Rs[idx_p] + 1) - (a1[idx_p] / (2 * Rs[idx_p])) * lambertwterm1 - \
                                 (a2[idx_p] / (2 * Rs[idx_p])) * lambertwterm2

            if output_is_scalar:
                return current.item()
            else:
                return current

    # finds the best solution and its fitness
    def result(self):
        fittest_index = np.argmin(self.fitnesses[-1])
        result = self.populations[-1][fittest_index]
        result_fitness = self.fitnesses[-1][fittest_index]
        assert result_fitness == np.min(self.fitnesses[-1][fittest_index])
        return result, result_fitness

    # PLOTTING
    # Plots a given vector solution
    def plot_solution(self, vector, vth):
        # Plot experimental points
        f, gr = plt.subplots()
        voltages, currents = self.ivdata
        xlim, ylim = max(voltages), max(currents)
        gr.plot(voltages, currents, 'go')

        # Calculate given solution adding 20% voltage axis length
        v = np.linspace(0, xlim + (.2 * xlim), 100)
        calc_current = DE.i_from_v(vector, v, vth, self.Ns, self.Np)

        # print("Solution:")
        # d = [print(str(i) + "\n") for i in zip(v, calc_current)]

        # Plotting (evil muuuhahahahaa)
        gr.title.set_text("IV characteristic")
        gr.plot(v, calc_current)
        gr.set_xlabel("Voltage V(V)")
        gr.set_ylabel("Current I(A)")
        gr.grid()
        gr.set_xlim((0, xlim + (.2 * xlim)))
        gr.set_ylim((0, ylim + (.2 * ylim)))
        plt.show()
        return f, gr

    def plot_fit_hist(self):
        assert self.maxiter == len(self.fitnesses)
        averages = [np.average(generation) for generation in self.fitnesses]
        # print("FITNESS HISTORY:")
        # [print(average) for average in averages]
        f, gr = plt.subplots()
        gr.title.set_text("Average population fitness")
        gr.plot(averages, 'b')
        gr.grid()
        gr.set_xlabel("Generation")
        gr.set_ylabel("RMSE")
        gr.set_xlim((0, self.maxiter - 1))
        plt.show()
        return f, gr

    # Plots the best solution vector
    def plot_result(self, temp):
        # plots the best solution
        vth = constants.Boltzmann * temp / constants.elementary_charge
        result, result_fitness = self.result()
        graph = self.plot_solution(result, vth)
        if self.dim == 5:
            print("Rsh = {}\nRs = {}\na = {}\nI0 = {}\nIpv = {}".format(*result))
        elif self.dim == 7:
            print("Rsh = {}\nRs = {}\na1 = {}\na2 = {}\nI01 = {}\nI02 = {}\nIpv = {}".format(*result))
        else:
            raise ValueError("Search space dimensionality must be either 5 for single or 7 for double diodes")
        print("Root Mean Squared Error:\t{0:.5E}".format(result_fitness))
        return graph


