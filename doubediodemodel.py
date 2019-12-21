import numpy as np
import scipy.constants as sc
import pvlib
import csv
import matplotlib.pyplot as plt

#%% CONTROL PARAMETERS & CONSTANTS
T = 51 + 273.15  # Temperature
Ns = 36  # Number of cells in series
Np = 1  # Number of cells in parallel
Vt = Ns * sc.Boltzmann * T / sc.elementary_charge  # Thermal voltage
GENMAX = 500  # Max number of generation
NP = 100  # Population size
F = 0.7  # Mutation factor (Values > 0.4 are recommended [1])
CR = 0.8  # Crossover rate (High values recommended [2])
D = 7  # 7-Dimensional search space

# Search space
rs_l, rs_h = 0, 0.5  # Series resistance
a1_l, a1_h = 1, 2  # Ideality factor boundaries
a2_l, a2_h = 1, 5
rp_l, rp_h = 10000, 30000  # Shunt resistance
ipv_l, ipv_h = 0, 10  # Photo-current
i0_l, i0_h = 0, 2  # Saturation current

fit_hist = []  # Store fitness history
avg_fit_hist = []


# Useful Functions

# J function calculation (RMSE)
def evaluate(a1, a2, rs, rp, ipv, i0):
    # Find the current using the Lambert W function [5]
    imodel = pvlib.pvsystem.i_from_v(rp, rs, a * Vt, voltages, i0, ipv)
    j = currents - imodel
    # Fitness penalty (J = 100) for unphysical values of Ipv, I0 and Rp
    if i0 < 0 or rp < 0 or ipv < 0:
        return 100

    return np.sqrt(np.mean(j ** 2))  # RMSE


with open("data/STM6_4036") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader)  # Ignore the header
    data = [(float(row[0]), float(row[1])) for row in csv_reader]
    voltages, currents = np.array([row[0] for row in data]), np.array([row[1] for row in data])

#%% PIVOT POINT SELECTION
p1, p2 = data[0], data[-1]  # Isc and Voc
# Maximum power point (MPP)
powers = [a * b for a, b in zip(voltages, currents)]
p3 = None
p = -1  # First MPP
p3_fit = []  # store fittest of each p3 run

# Population initialization
gen = 1

while p <= 1:

    p3 = data[powers.index(max(powers)) + p]  # P(-1), P(0) and P(1)
    POP = np.random.uniform([a1_l, a2_l, rs_l, rp_l, ipv_l, i0_l, i0_l],
                            [a1_h, a2_l, rs_h, rp_h, ipv_h, i0_h, i0_h], size=(NP, D))  # Population initialization

    while gen <= GENMAX:

        prevPOP = POP
        Jrand = np.random.randint(0, NP)  # Jrand

        fitness = np.asarray([evaluate(*vec, p3) for vec in POP])
        fittest_index = np.argmin(fitness)
        fittest = POP[fittest_index]
        fit_hist.append(fitness[fittest_index])
        avg_fit_hist.append(np.average(fitness))

        for i in range(NP):
            a1, a2, rs, rp, ipv, i01, i02 = POP[i]
            # Generate donor vector from 2 random vectors and the fittest
            indices = [ind for ind in range(NP) if ind != i and ind != fittest_index]
            i1, i2 = np.random.choice(indices, 2, replace=False)
            # DE mutation
            mutant = fittest + F * (POP[i1] - POP[i2])
            # Crossover
            crosspoints = np.random.rand(D) <= CR  # List of booleans
            if not any(crosspoints):
                crosspoints[np.random.randint(0, D)] = True  # Force at least one crosspoint
            trial = np.where(crosspoints, mutant, POP[i])
            # Penalty
            # First ideality factor
            if not a1_l <= trial[0]:
                trial[0] = a1_l + np.random.rand() * (a1_h - a1_l)
            if not a1_h >= trial[0]:
                trial[0] = a1_h - np.random.rand() * (a1_h - a1_l)
            # Second ideality factor
            if not a2_l <= trial[1]:
                trial[1] = a2_l + np.random.rand() * (a2_h - a2_l)
            if not a2_h >= trial[1]:
                trial[1] = a2_h - np.random.rand() * (a2_h - a2_l)
            # Series resistance
            if not rs_l <= trial[2]:
                trial[2] = rs_l + np.random.rand() * (rs_h - rs_l)
            if not rs_h >= trial[2]:
                trial[2] = rs_h - np.random.rand() * (rs_h - rs_l)
            # Shunt resistance
            if not rp_l <= trial[3]:
                trial[3] = rp_l + np.random.rand() * (rp_h - rp_l)
            if not rp_h >= trial[3]:
                trial[3] = rp_h - np.random.rand() * (rp_h - rp_l)
            # Photo-current
            if not ipv_l <= trial[4]:
                trial[4] = ipv_l + np.random.rand() * (ipv_h - ipv_l)
            if not ipv_h >= trial[4]:
                trial[4] = ipv_h - np.random.rand() * (ipv_h - ipv_l)
            # Saturation currents
            if not i0_l <= trial[5]:
                trial[5] = i0_l + np.random.rand() * (i0_h - i0_l)
            if not i0_h >= trial[5]:
                trial[5] = i0_h - np.random.rand() * (i0_h - i0_l)
            if not i0_l <= trial[6]:
                trial[6] = i0_l + np.random.rand() * (i0_h - i0_l)
            if not i0_h >= trial[6]:
                trial[6] = i0_l - np.random.rand() * (i0_h - i0_l)

            # Selection
            f = evaluate(*trial, p3)
            if f < fitness[i]:
                fitness[i] = f
                POP[i] = trial  # Switch
                # Movie plotting
                # plt.plot(voltages, currents, 'go')
                # v = np.linspace(0, 25, 100)
                # rp, rs, a, i0, ipv = get_5_from_2(a, Rs, p3)
                a1, a2, rs, rp, ipv, i01, i02 = *trial
                ical = pvlib.
                plt.plot(v, ical)
                plt.xlabel("Voltage V(V)")
                plt.ylabel("Current I(A)")
                plt.grid()
                plt.axis([0, 25, 0, 2])
                camera.snap()

                if f < fitness[fittest_index]:
                    fittest_index = i
                    fittest = trial

        gen += 1
    animation = camera.animate()
    animation.save("evol{}-{}.mp4".format(p, gen))

    plot_from_2(*fittest, p3)
    p3_fit.append((*fittest, p3))
    p += 1

