import csv
import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
import pvlib
from scipy.special import lambertw
import decimal

T = 51 + 273.15  # Temperature
Ns = 36  # Number of cells in series
Np = 1  # Number of cells in parallel
Vt = (sc.Boltzmann * T) / sc.elementary_charge  # Thermal voltage


with open("data/STM6_4036") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader)  # Ignore the header
    data = [(float(row[0]), float(row[1])) for row in csv_reader]
    voltages, currents = np.array([row[0] for row in data]), np.array([row[1] for row in data])
    N = len(voltages)


# def i_from_v(resist_sh, resist_series, n, voltage, sat_current, photocurrent):
#     conductance_shunt = 1. / resist_sh
#     I = np.zeros_like(voltage)
#     if resist_series == 0:
#         i = photocurrent - sat_current * np.expm1(voltage / (n * Vt)) - (voltage / resist_sh)
#     else:
#         argW = resist_series * sat_current * np.exp((resist_series * (photocurrent + sat_current) + v)
#                                                 / (n * Vt * (resist_series * conductance_shunt + 1.))) \
#                / (n * Vt * (resist_series * conductance_shunt + 1.))
#
#         wterm = lambertw(argW).real
#
#         i = (sat_current + photocurrent - conductance_shunt * v) / (resist_sh * conductance_shunt + 1.)\
#             - (n * Vt / resist_series) * wterm
#         print(i)
#     return i


# STP6
# rp = 10.5309
# rs = 5.3819e-3
# a = 1.1872
# i0 = 0.8868e-6
# ipv = 7.4830

# STM6
rp = 16.7328
rs = 0.4186e-3
a = 1.5656
i0 = 2.7698e-6
ipv = 1.6632

# RTC France
# rp = 54.19241
# rs = 0.03629
# a = 1.47986
# i0 = 0.31911e-6
# ipv = 0.76072

# PWP201
# rp = 27.27729
# rs = 0.03337
# a = 1.35119
# i0 = 3.48226e-6
# ipv = 1.03051
p3 = data[10]
p1 = data[0]
p2 = data[-1]


def get_5_from_2(a, Rs, p3):
    # Finds the 5 single diode model params from a, Rs, p1, p2 and p3
    # Re-extract the 5 parameters from each solution vector by formfitting on the pivot points
    alpha = p3[0] - p1[0] + (Rs * (p3[1] - p1[1]) * (Ns / Np))
    beta = p2[0] - p1[0] + (Rs * (p2[1] - p1[1]) * (Ns / Np))
    T = 51 + 273.15  # Temperature
    loc_Vt = (Ns * sc.Boltzmann * T) / sc.elementary_charge

    i0 = (alpha * (p2[1] - p1[1]) + beta * (p3[1] - p1[1])) / (Np * (alpha * (
            np.exp((p1[0] + p1[1] * Rs * (Ns / Np)) / (a * loc_Vt)) -
            np.exp((p2[0] + p2[1] * Rs * (Ns / Np)) / (a * loc_Vt))) + beta * (
            np.exp((p1[0] + p1[1] * Rs * (Ns / Np)) / (a * loc_Vt)) -
            np.exp((p3[0] + p3[1] * Rs * (Ns / Np)) / (a * loc_Vt)))))

    rp = ((p1[0] - p2[0]) * (Np / Ns) + Rs * (p1[1] - p2[1])) / (p2[1] - p1[1] - i0 * Np * (
            np.exp((p1[0] + p1[1] * Rs * (Ns / Np)) / (a * loc_Vt)) -
            np.exp((p2[0] + p2[1] * Rs * (Ns / Np)) / (a * loc_Vt))))

    ipv = (i0 * Np * (np.exp((p1[0] + p1[1] * Rs * (Ns / Np)) / (a * loc_Vt)) - 1) +
           ((p1[0] + p1[1] * Rs * (Ns / Np)) / (rp * (Ns / Np))) + p1[1]) * (1 / Np)

    return rp, Rs, a, i0, ipv


print(get_5_from_2(a, rs, p3))

v = np.linspace(0, 25, 100)
# ical = i_from_v(rp * Ns, rs, a * Ns * Vt, v, i0, ipv)
ical = pvlib.pvsystem.i_from_v(rp * Ns, rs, a * Ns * Vt, v, i0, ipv)
plt.plot(v, ical)
plt.plot(voltages, currents, 'go')
plt.grid()
plt.axis([0, 23, 0, 2])
plt.show()
exit(0)
