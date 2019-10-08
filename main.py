import csv
import numpy as np
from helpers import *

# CONTROL PARAMETERS
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
    voltages, currents = np.array([row[0] for row in csv_reader]), np.array([row[1] for row in csv_reader])
    if sorted(voltages) != voltages:
        print("DATA NOT SORTED FROM LOWEST TO HIGHEST VOLTAGES")

# PIVOT POINT SELECTION
p1, p2 = data[1], data[-1]  # Isc and Voc
# Maximum power point (MPP)
p3 = get_mpp(data)

# Population initialization
gen = 1
POP = np.random.uniform([a_L, Rs_L], [a_H, Rs_H], size=(NP, 2))
POP = np.hstack((POP, np.full((NP, 1), 100)))  # Fitness value on the third column

while gen <= GENMAX:
    # fitness penalty (J = 100) for unphysical values of Ipv, I0 and Rp
    Vt = 26e-3
    # Re-extract the 5 parameters from each solution vector by formfitting on the pivot points
    for line in range(POP.shape[0]):
        a, Rs = POP[line,0], POP[line,1]
        # print(a.shape, Rs.shape)

        # Behold..#######################################################################
        alpha = p3[0] - p1[0] + (Rs * (p3[1] - p1[1]))
        beta = p2[0] - p1[0] + (Rs * (p2[1] - p1[1]))

        I0 = (alpha * (p2[1] - p1[1]) + beta * (p3[1] - p1[1])) / \
             (alpha * (np.exp((p1[0] + p1[1] * Rs) / (a * Vt)) - np.exp((p2[0] - p2[1] * Rs) / (a * Vt))) +
              beta * (np.exp((p1[0] + p1[1] * Rs) / (a * Vt)) - np.exp((p3[0] + p3[1] * Rs) / (a * Vt))))

        Rp = ((p1[0] - p2[0]) + Rs * (p1[1] - p2[1])) / (p2[1] - p1[1] - I0 * (
                np.exp((p1[0] + p1[1] * Rs) / (a * Vt)) - np.exp((p2[0] + p2[1] * Rs) / (a * Vt))))

        Ipv = I0 * (np.exp((p1[0] + p1[1] * Rs) / (a * Vt)) - 1) + ((p1[0] + p1[1] * Rs) / Rp) + p1[1]
        ##################################################################################
        if I0 < 0 or Rp < 0 or Ipv < 0:
            POP[line:2] = 100
        else:
            # J function calculation
            # I = Ipv - (I0 * np.exp((voltages +) / )

    gen += 1

print("done")
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