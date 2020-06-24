import csv
import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
# import pvlib
from scipy.special import lambertw
import objects

k = sc.Boltzmann
q = sc.elementary_charge
temp_c = 55
T = temp_c + 273.15  # Temperature
Ns = 1  # Number of cells in series
Np = 1  # Number of cells in parallel
Vt = (k * T) / q  # Thermal voltage


# with open("data/STP6") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=",")
#     next(csv_reader)  # Ignore the header
#     data = [(float(row[0]), float(row[1])) for row in csv_reader]
#     voltages, currents = np.array([row[0] for row in data]), np.array([row[1] for row in data])
#     N = len(voltages)
#     voltages, currents = np.array([row[0] for row in data]), np.array([row[1] for row in data])


def i_from_v(resistance_shunt, resistance_series, n, voltage, saturation_current, photocurrent):
    nNsVth = n * Ns * Vt
    output_is_scalar = all(map(np.isscalar,
                               [resistance_shunt, resistance_series, nNsVth,
                                voltage, saturation_current, photocurrent]))

    # This transforms Gsh=1/Rsh, including ideal Rsh=np.inf into Gsh=0., which
    #  is generally more numerically stable
    conductance_shunt = 1. / resistance_shunt

    # Ensure that we are working with read-only views of numpy arrays
    # Turns Series into arrays so that we don't have to worry about
    #  multidimensional broadcasting failing
    Gsh, Rs, a, V, I0, IL = \
        np.broadcast_arrays(conductance_shunt, resistance_series, nNsVth,
                            voltage, saturation_current, photocurrent)

    # Intitalize output I (V might not be float64)
    I = np.full_like(V, np.nan, dtype=np.float64)  # noqa: E741, N806

    # Determine indices where 0 < Rs requires implicit model solution
    idx_p = 0. < Rs

    # Determine indices where 0 = Rs allows explicit model solution
    idx_z = 0. == Rs

    # Explicit solutions where Rs=0
    if np.any(idx_z):
        I[idx_z] = IL[idx_z] - I0[idx_z] * np.expm1(V[idx_z] / a[idx_z]) - \
                   Gsh[idx_z] * V[idx_z]

    # Only compute using LambertW if there are cases with Rs>0
    # Does NOT handle possibility of overflow, github issue 298
    if np.any(idx_p):
        # LambertW argument, cannot be float128, may overflow to np.inf
        argW = Rs[idx_p] * I0[idx_p] / (
                a[idx_p] * (Rs[idx_p] * Gsh[idx_p] + 1.)) * \
               np.exp((Rs[idx_p] * (IL[idx_p] + I0[idx_p]) + V[idx_p]) /
                      (a[idx_p] * (Rs[idx_p] * Gsh[idx_p] + 1.)))

        # lambertw typically returns complex value with zero imaginary part
        # may overflow to np.inf
        lambertwterm = lambertw(argW).real

        # Eqn. 2 in Jain and Kapoor, 2004
        #  I = -V/(Rs + Rsh) - (a/Rs)*lambertwterm + Rsh*(IL + I0)/(Rs + Rsh)
        # Recast in terms of Gsh=1/Rsh for better numerical stability.
        I[idx_p] = (IL[idx_p] + I0[idx_p] - V[idx_p] * Gsh[idx_p]) / \
                   (Rs[idx_p] * Gsh[idx_p] + 1.) - (
                           a[idx_p] / Rs[idx_p]) * lambertwterm

    if output_is_scalar:
        return I.item()
    else:
        return I


# STP6
# rp = 10.5309
# rs = 5.3819e-3
# a = 1.1872
# i0 = 0.8868e-6
# ipv = 7.4830

# STM6
# rp = 16.7328
# rs = 0.4186e-3
# a = 1.5656
# i0 = 2.7698e-6
# ipv = 1.6632
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

#
# p3 = data[7]
# p1 = data[0]
# p2 = data[-1]
#
#
# def get_5_from_2(a, rs, p3):
#     # Finds the 5 single diode model params from a, rs, p1, p2 and p3
#     # Re-extract the 5 parameters from each solution vector by formfitting on the pivot points
#
#     v3, v2, v1 = p3[0], p2[0], p1[0]
#     i3, i2, i1 = p3[1], p2[1], p1[1]
#     t = temp_c + 273.15  # Temperature
#     loc_vt = (Ns * k * t) / q
#
#     alpha = v3 - v1 + (rs * (i3 - i1) * (Ns / Np))
#     beta = v2 - v1 + (rs * (i2 - i1) * (Ns / Np))
#
#     i0 = (alpha * (i2 - i1) + beta * (i3 - i1)) / (Np * (alpha * (
#             np.exp((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) -
#             np.exp((v2 + i2 * rs * (Ns / Np)) / (a * loc_vt))) + beta * (
#             np.exp((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) -
#             np.exp((v3 + i3 * rs * (Ns / Np)) / (a * loc_vt)))))
#
#     rp = -((v1 - v2) * (Np / Ns) + rs * (i1 - i2)) / (i2 - i1 - i0 * Np * (
#             np.exp((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) -
#             np.exp((v2 + i2 * rs * (Ns / Np)) / (a * loc_vt))))
#
#     ipv = (i0 * Np * np.expm1((v1 + i1 * rs * (Ns / Np)) / (a * loc_vt)) +
#            ((v1 + i1 * rs * (Ns / Np)) / (rp * (Ns / Np))) + i1) * (1 / Np)
#
#     return rp, rs, a, i0, ipv
#
#
# print(p3)
# print(get_5_from_2(a, rs, p3))
#
# v = np.linspace(0, 25, 100)
# ical = i_from_v(rp * Ns / Np, rs * Ns / Np, a * Ns * Vt, v, i0 * Np, ipv * Np)
# plt.plot(v, ical)
# plt.plot(voltages, currents, 'go')
# plt.grid()
# # plt.axis([0, 25, 0, 10])
# plt.show()

# rs = 0.001
# rs2 = 0.03
# rs3 = 0.1
# rp = 10
# a = 1
# i0 = 1e-6
# ipv = 5
# v = np.linspace(0, 0.5, 100)
# ical = i_from_v(rp, rs, a, v, i0, ipv)
# ical2 = i_from_v(rp, rs2, a, v, i0, ipv)
# ical3 = i_from_v(rp, rs3, a, v, i0, ipv)
#
# plt.plot(v, ical, label=r"$R_S = 1m\Omega$")
# plt.plot(v, ical2, label=r"$R_S = 30m\Omega$")
# plt.plot(v, ical3, label=r"$R_S = 0.1\Omega$")
# plt.legend()
# plt.grid()
# plt.xlabel("V(V)")
# plt.ylabel("I(A)")
# plt.axis([0, 0.5, 0, 5.2])
# plt.show()

# z = np.linspace(-1/np.e, 3, 100000)
# zm1 = np.linspace(-1/np.e, 0, 10000)
# w0 = lambertw(z, 0).real
# wm1 = lambertw(zm1, -1).real
# fig, ax = plt.subplots()
# ax.plot(z, w0, label=r"$W_0(z)$")
# ax.plot(zm1, wm1, label=r"$W_{-1}(z)$")
# ax.axis([-1, 4, -4, 1])
# ax.axhline(y=0, color='k')
# ax.axvline(x=0, color='k')
# ax.legend()
# plt.ylabel(r'$W(x)$')
# plt.xlabel(r'$x$')
# plt.grid()
# plt.show()

# RAW DATA PLOTTING
# v, i = objects.read_csv("/home/youssef/PycharmProjects/DEPV/data/RTC33D1000W.csv")
# plt.plot(v, i, 'go')
# plt.title("Points expérimentaux")
# plt.xlabel("Voltage V(V)")
# plt.ylabel("Courant I(A)")
# plt.axis([0, 0.7, 0, 0.9])
# plt.grid()
# plt.show()