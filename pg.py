import csv
import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
import pvlib
from scipy.special import lambertw

k = 1.3806503e-23
q = 1.60217646e-19
temp_c = 51
T = temp_c + 273.15  # Temperature
Ns = 36  # Number of cells in series
Np = 1  # Number of cells in parallel
Vt = (k * T) / q  # Thermal voltage


with open("data/STM6_4036") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader)  # Ignore the header
    data = [(float(row[0]), float(row[1])) for row in csv_reader]
    voltages, currents = np.array([row[0] for row in data]), np.array([row[1] for row in data])
    N = len(voltages)
    voltages, currents = np.array([row[0] for row in data]), np.array([row[1] for row in data])



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
# rp, rs, a, i0, ipv = 25.798932693041625, 0.0004186, 1.5656, 3.0506095606306254e-06, 1.6631543240678817

# RTC France
# rp = 54.19241
# rs = 0.03629
# a = 1.47986
# i0 = 0.31911e-6
# ipv = 0.76072

# PWP201
# rp = 19.37196
# rs = 0.03471
# a = 1.30017
# i0 = 2.12479e-6
# ipv = 1.03353


p3 = data[11]
p1 = data[0]
p2 = data[-1]


def get_5_from_2(a, rs, p3):
    # Finds the 5 single diode model params from a, rs, p1, p2 and p3
    # Re-extract the 5 parameters from each solution vector by formfitting on the pivot points

    v3, v2, v1 = p3[0], p2[0], p1[0]
    i3, i2, i1 = p3[1], p2[1], p1[1]
    t = temp_c + 273.15  # Temperature
    loc_vt = (Ns * k * t) / q

    alpha = v3 - v1 + (rs * (i3 - i1) * (Ns / Np))
    beta = v2 - v1 + (rs * (i2 - i1) * (Ns / Np))

    i0 = (alpha * (i2 - i1) + beta * (i3 - i1)) / (Np * (alpha * (
            np.exp((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) -
            np.exp((v2 + i2 * rs * (Ns / Np)) / (a * loc_vt))) + beta * (
            np.exp((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) -
            np.exp((v3 + i3 * rs * (Ns / Np)) / (a * loc_vt)))))

    rp = -((v1 - v2) * (Np / Ns) + rs * (i1 - i2)) / (i2 - i1 - i0 * Np * (
            np.exp((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) -
            np.exp((v2 + i2 * rs * (Ns / Np)) / (a * loc_vt))))

    ipv = (i0 * Np * np.expm1((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) +
           ((v1 + i1 * rs * (Ns / Np)) / (rp * (Ns / Np))) + i1) * (1 / Np)

    return rp, rs, a, i0, ipv


print(p3)
print(get_5_from_2(a, rs, p3))

v = np.linspace(0, 25, 100)
# ical = i_from_v(rp * Ns, rs, a * Ns * Vt, v, i0, ipv)
ical = pvlib.pvsystem.i_from_v(rp * Ns / Np, rs * Ns / Np, a * Ns * Vt, v, i0 * Np, ipv * Np)
plt.plot(v, ical)
plt.plot(voltages, currents, 'go')
plt.grid()
plt.axis([0, 25, 0, 2])
plt.show()
exit(0)
