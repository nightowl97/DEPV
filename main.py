import csv
import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
import pvlib
import decimal
from celluloid import Camera


#%% CONTROL PARAMETERS & CONSTANTS
temp_c = 55  # Temperature in celsius
T = temp_c + 273.15  # Temperature
Ns = 36  # Number of cells in series
Np = 1  # Number of cells in parallel
Vt = sc.Boltzmann * T / sc.elementary_charge  # Thermal voltage
GENMAX = 100  # Max number of generations
NP = 100  # Population size
F = 0.7  # Mutation factor (Values > 0.4 are recommended [1])
CR = 0.8  # Crossover rate (High values recommended [2])
D = 2  # 2-Dimensional search space (a, Rs)
# Search space according to literature ([3] Oliva et al., 2017 and [4] Yu et al. 2017)
a_L, a_H = 1, 2  # Ideality factor boundaries
Rs_L, Rs_H = 0, 0.5  # Series resistance Rs
fit_hist = []  # Store fitness history
avg_fit_hist = []

#%% DATA COLLECTION
# CSV file
with open("data/STP6") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader)  # Ignore the header
    data = [(float(row[0]), float(row[1])) for row in csv_reader]
    voltages, currents = np.array([row[0] for row in data]), np.array([row[1] for row in data])
    N = len(voltages)
    xlim = max(voltages) + int(.1 * max(voltages))
    ylim = max(currents) + int(.1 * max(currents)) + .02


#%% PIVOT POINT SELECTION
p1, p2 = data[0], data[-1]  # Isc and Voc
# Maximum power point (MPP)
powers = [a * b for a, b in zip(voltages, currents)]
p3 = data[powers.index(max(powers))]
p = -2  # First MPP
p3_fit = []  # store fittest of each p3 run


def get_5_from_2(a, rs, mpp):
    # Finds the 5 single diode model params from a, rs, p1, p2 and p3
    # Re-extract the 5 parameters from each solution vector by formfitting on the pivot points

    v3, v2, v1 = mpp[0], p2[0], p1[0]
    i3, i2, i1 = mpp[1], p2[1], p1[1]
    t = temp_c + 273.15  # Temperature
    loc_vt = (Ns * sc.Boltzmann * t) / sc.elementary_charge

    alpha = v3 - v1 + (rs * (i3 - i1) * (Ns / Np))
    beta = v2 - v1 + (rs * (i2 - i1) * (Ns / Np))

    i0 = (alpha * (i2 - i1) + beta * (i3 - i1)) / (Np * (alpha * (
            np.exp((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) -
            np.exp((v2 + i2 * rs * (Ns / Np)) / (a * loc_vt))) + beta * (
            np.exp((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) -
            np.exp((v3 + i3 * rs * (Ns / Np)) / (a * loc_vt)))))

    rp = ((v1 - v2) * (Np / Ns) + rs * (i1 - i2)) / (i2 - i1 - i0 * Np * (
            np.exp((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) -
            np.exp((v2 + i2 * rs * (Ns / Np)) / (a * loc_vt))))

    ipv = (i0 * Np * np.expm1((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) +
           ((v1 + i1 * rs * (Ns / Np)) / (rp * (Ns / Np))) + i1) * (1 / Np)

    return rp, rs, a, i0, ipv


# Evaluation function using the three pivot points
def evaluate(a, Rs, p3):
    # J function calculation (RMSE)
    # Find the current using the Lambert W function [5]
    Rp, Rs, a, I0, Ipv = get_5_from_2(a, Rs, p3)
    Ical = pvlib.pvsystem.i_from_v(Rp * Ns / Np, Rs * Ns / Np, a * Ns * Vt, voltages, I0 * Np, Ipv * Np)
    J = currents - Ical
    # Fitness penalty (J = 100) for unphysical values of Ipv, I0 and Rp
    if I0 < 0 or 1000 < Rp or Rp < 0 or Ipv < 0:
        return 100

    return np.sqrt(np.mean(J ** 2))  # RMSE


def plot_from_2(a, rseries, mpp):
    # Plots I-V curve from the two parameters a, Rs
    plt.plot(voltages, currents, 'go')
    v = np.linspace(0, xlim, 100)
    rp, rs, a, i0, ipv = get_5_from_2(a, rseries, mpp)
    ical = pvlib.pvsystem.i_from_v(rp * Ns / Np, rs * Ns / Np, a * Ns * Vt, v, i0 * Np, ipv * Np)
    plt.plot(v, ical)
    plt.xlabel("Voltage V(V)")
    plt.ylabel("Current I(A)")
    plt.grid()
    plt.axis([0, xlim, 0, ylim])
    plt.show()
    plt.figure()


# fig = plt.figure()
# camera = Camera(fig)
while p <= 2:

    gen = 1
    p3 = data[powers.index(max(powers)) + p * int(.1 * N)]  # P(-1), P(0) and P(1)
    POP = np.random.uniform([a_L, Rs_L], [a_H, Rs_H], size=(NP, 2))  # Population initialization

    while gen <= GENMAX:

        fitness = np.asarray([evaluate(vec[0], vec[1], p3) for vec in POP])
        fittest_index = np.argmin(fitness)
        fittest = POP[fittest_index]
        fit_hist.append(fitness[fittest_index])
        avg_fit_hist.append(np.average(fitness))

        for i in range(NP):
            a, Rs = POP[i]
            # Generate donor vector from 2 random vectors and the fittest
            indices = [ind for ind in range(NP) if ind != i and ind != fittest_index]
            i1, i2 = np.random.choice(indices, 2, replace=False)
            # DE mutation
            mutant = fittest + F * (POP[i1] - POP[i2])
            # Crossover
            crosspoints = np.random.rand(D) <= CR
            if not any(crosspoints):
                trial = np.where(np.random.randint(0, D), mutant, POP[i])
            else:
                trial = np.where(crosspoints, mutant, POP[i])
            # Penalty
            if not a_L <= trial[0]:
                trial[0] = a_L + np.random.rand() * (a_H - a_L)
            if not a_H >= trial[0]:
                trial[0] = a_H - np.random.rand() * (a_H - a_L)
            if not Rs_L <= trial[1]:
                trial[1] = Rs_L + np.random.rand() * (Rs_H - Rs_L)
            if not Rs_H >= trial[1]:
                trial[1] = Rs_H - np.random.rand() * (Rs_H - Rs_L)

            # Selection
            f = evaluate(trial[0], trial[1], p3)
            if f < fitness[i]:
                fitness[i] = f
                POP[i] = trial  # Switch
                # Movie plotting
                # plt.plot(voltages, currents, 'go')
                # v = np.linspace(0, xlim, 100)
                # rp, rs, a, i0, ipv = get_5_from_2(a, Rs, p3)
                # ical = pvlib.pvsystem.i_from_v(rp * Ns / Np, rs * Ns / Np, a * Ns * Vt, v, i0 * Np, ipv * Np)
                # plt.plot(v, ical)
                # plt.xlabel("Voltage V(V)")
                # plt.ylabel("Current I(A)")
                # plt.grid()
                # plt.axis([0, xlim, 0, ylim])
                # camera.snap()

                if f < fitness[fittest_index]:
                    fittest_index = i
                    fittest = trial

        gen += 1
    # animation = camera.animate()
    # animation.save("evol{}-{}.mp4".format(p, gen))

    plot_from_2(*fittest, p3)
    p3_fit.append((*fittest, p3))
    p += 1

fittest = p3_fit[np.argmin([evaluate(*e) for e in p3_fit])]


print("MPP: {}".format(fittest))
print("RESULTS:\n(Rp:{}, Rs:{}, a:{}, I0:{}, Ipv:{})".format(*get_5_from_2(*fittest)))
print("RMSE: {:.2E}".format(decimal.Decimal(evaluate(*fittest))))
# print(get_5_from_2(1.5656, 0.4186e-3, (16.4, 1.542)))
# print(evaluate(1.5656, 0.4186e-3, (16.4, 1.542)))

# DATA REPRESENTATION
plot_from_2(*fittest)

plt.plot(avg_fit_hist, 'b')
plt.grid()
plt.xlabel("Generation")
plt.ylabel("RMSE")
plt.xlim((0, GENMAX - 1))
plt.show()
exit(0)

"""
References
[1] Price K, Storn RM, Lampinen JA. Differential evolution: a practical
approach to global optimization. Springer Science & Business Media; 2006 Mar 4.
[2]  Wei H, Cong J, Lingyun X, Deyun S. Extracting solar cell model parameters 
based on chaos particle swarm algorithm. In2011 International Conference on 
Electric Information and Control Engineering 2011 Apr 15 (pp. 398-402). IEEE.
URL: https://ieeexplore.ieee.org/abstract/document/5777246/
[3] Yu K, Chen X, Wang X, Wang Z. Parameters identification of photovoltaic 
models using self-adaptive teaching-learning-based optimization. Energy 
Conversion and Management. 2017 Aug 1;145:233-46.
URL: https://www.sciencedirect.com/science/article/pii/S0306261917305330
[4] Oliva D, El Aziz MA, Hassanien AE. Parameter estimation of photovoltaic 
cells using an improved chaotic whale optimization algorithm. 
Applied Energy. 2017 Aug 15;200:141-54.
URL: https://www.sciencedirect.com/science/article/pii/S0196890417303655
[5] A. Jain, A. Kapoor, "Exact analytical solutions of the
parameters of real solar cells using Lambert W-function", Solar
Energy Materials and Solar Cells, 81 (2004) 269-277.
URL: https://www.sciencedirect.com/science/article/pii/S0927024803002605
"""