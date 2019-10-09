import csv
import random
import matplotlib.pyplot as plt

import numpy as np

# CONTROL PARAMETERS & CONSTANTS
Vt = 26e-3 # Thermal voltage
GENMAX = 50  # Max number of generation
NP = 50  # Population size
F = 0.7  # Mutation factor (Values > 0.4 are recommended [1])
CR = 0.8  # Crossover rate (High values recommended [2])
D = 2  # 2-Dimensional search space (a, Rs)
# Search space according to literature ([3] Oliva et al., 2017 and [4] Yu et al. 2017)
a_L, a_H = 1, 2  # Ideality factor boundaries
Rs_L, Rs_H = 0, 0.5  # Series resistance Rs
# TODO: add Ns and Np for modules and to calculate thermal voltage, use Vt = 26mV for now

# DATA COLLECTION
with open("data/RTC33D1000W.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader)  # Ignore the header
    data = [(float(row[0]), float(row[1])) for row in csv_reader]
    voltages, currents = np.array([row[0] for row in data]), np.array([row[1] for row in data])

# PIVOT POINT SELECTION
p1, p2 = data[1], data[-1]  # Isc and Voc
# Maximum power point (MPP)
powers = [a * b for a, b in zip(voltages, currents)]
p3 = data[powers.index(max(powers))]


# Evaluation function using the three pivot points
def evaluate(a, Rs):
    # TODO: Double check Ical
    # J function calculation
    # Re-extract the 5 parameters from each solution vector by formfitting on the pivot points
    alpha = p3[0] - p1[0] + (Rs * (p3[1] - p1[1]))
    beta = p2[0] - p1[0] + (Rs * (p2[1] - p1[1]))

    I0 = (alpha * (p2[1] - p1[1]) + beta * (p3[1] - p1[1])) / \
         (alpha * (np.exp((p1[0] + p1[1] * Rs) / (a * Vt)) - np.exp((p2[0] - p2[1] * Rs) / (a * Vt))) +
          beta * (np.exp((p1[0] + p1[1] * Rs) / (a * Vt)) - np.exp((p3[0] + p3[1] * Rs) / (a * Vt))))

    Rp = ((p1[0] - p2[0]) + Rs * (p1[1] - p2[1])) / (p2[1] - p1[1] - I0 * (
            np.exp((p1[0] + p1[1] * Rs) / (a * Vt)) - np.exp((p2[0] + p2[1] * Rs) / (a * Vt))))

    Ipv = I0 * (np.exp((p1[0] + p1[1] * Rs) / (a * Vt)) - 1) + ((p1[0] + p1[1] * Rs) / Rp) + p1[1]

    Ical = Ipv - (I0 * np.exp((voltages + currents * Rs) / (a * Vt)) - 1) - ((voltages + currents * Rs) / Rp)

    # Fitness penalty (J = 100) for unphysical values of Ipv, I0 and Rp
    if I0 < 0 or Rp < 0 or Ipv < 0:
        return 100, Ical

    return np.sqrt(np.mean((currents - Ical) ** 2)), Ical  # RMSE


# Population initialization
gen = 1
POP = np.random.uniform([a_L, Rs_L], [a_H, Rs_H], size=(NP, 2))


while gen <= GENMAX:

    prevPOP = POP
    Jrand = np.random.randint(0, NP)  # Jrand

    fitness = np.asarray([evaluate(vec[0], vec[1])[0] for vec in POP])
    fittest_index = np.argmin(fitness)
    fittest = POP[fittest_index]

    for i in range(NP):
        a, Rs = POP[i, 0], POP[i, 1]
        # Generate donor vector from 2 random vectors and the fittest
        indices = [ind for ind in range(NP) if ind != i and ind != fittest_index]
        i1, i2 = np.random.choice(indices, 2, replace=False)
        # DE mutation
        mutant = fittest + F * (POP[i1] - POP[i2])
        # Crossover
        trial = np.where(np.random.rand(2) <= CR, mutant, POP[i])  # TODO: Guarantee at least one crossover
        # Penalty
        if not a_L <= trial[0]: trial[0] = a_L + np.random.rand() * (a_H - a_L)
        if not a_H >= trial[0]: trial[0] = a_H - np.random.rand() * (a_H - a_L)
        if not Rs_L <= trial[1]: trial[1] = Rs_L + np.random.rand() * (Rs_H - Rs_L)
        if not Rs_H >= trial[1]: trial[1] = Rs_H - np.random.rand() * (Rs_H - Rs_L)

        # Selection
        f = evaluate(trial[0], trial[1])[0]
        if f < fitness[i]:
            fitness[i] = f
            POP[i] = trial  # Switch
            if f < fitness[fittest_index]:
                fittest_index = i
                fittest = trial

    print("Global fitness: {}".format(sum(fitness)))
    gen += 1
    plt.plot(voltages, currents, 'bo')
    plt.plot(voltages, evaluate(fittest[0], fittest[1])[1])
    # plt.axis([0, .6, 0, .8])
    plt.show()

print("RESULTS:\t|a\t\t\t\t\t|Rs\nPARAMS:\t\t|{}\t|{}".format(fittest[0], fittest[1]))

# DATA REPRESENTATION
# I-V Characteristic

plt.plot(voltages, currents, 'bo')
plt.plot(voltages, evaluate(fittest[0], fittest[1])[1])
# plt.axis([0, .6, 0, .8])
plt.show()

exit(0)
"""
References
[1]
URL: 
[2]
URL: 
[3] Yu K, Chen X, Wang X, Wang Z. Parameters identification of photovoltaic 
models using self-adaptive teaching-learning-based optimization. Energy 
Conversion and Management. 2017 Aug 1;145:233-46.
URL: https://www.sciencedirect.com/science/article/pii/S0306261917305330
[4] Oliva D, El Aziz MA, Hassanien AE. Parameter estimation of photovoltaic 
cells using an improved chaotic whale optimization algorithm. 
Applied Energy. 2017 Aug 15;200:141-54.
URL: https://www.sciencedirect.com/science/article/pii/S0196890417303655
"""