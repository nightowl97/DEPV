import matplotlib.pyplot as plt
import numpy as np
from testing import i_from_v
from objects import DE, read_csv
import scipy.constants as constants
from matplotlib import cm

# Experimental data RTC France
expdata = read_csv("data/RTC33D1000W.csv")

rsh = 54.1134
i0 = 3.209e-07
ipv = 0.7607
rsgrid = np.arange(0.036, 0.0368, 0.000001)
agrid = np.arange(1.4705, 1.4715, 0.00001)
temp = 33 + 275.15
vth = constants.Boltzmann * temp / constants.elementary_charge
z = np.zeros((rsgrid.size, agrid.size))

for x in range(rsgrid.size):
    for y in range(agrid.size):
        v = np.asarray([rsh, rsgrid[x], agrid[y], i0, ipv])
        z[x, y] = DE.objf(v, expdata, vth, 1, 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
RS, A = np.meshgrid(rsgrid, agrid)
Z = z.reshape(RS.shape)

ax.plot_surface(RS, A, Z, cmap='viridis')
# ax.set_zlim(0, 6)
ax.set_xlabel(r"$R_s$")
ax.set_ylabel(r"$a$")
ax.set_zlabel(r"$RMSE$")

plt.show()

# RS
# rsh = 54.1134
# i0 = 3.209e-07
# ipv = 0.7607
# rsgrid = np.arange(0.01, 0.1, 0.001)
# a = 1.4709
# temp = 33 + 275.15
# vth = constants.Boltzmann * temp / constants.elementary_charge
#
# rms = np.zeros(rsgrid.shape)
#
# for x in range(rsgrid.size):
#     v = np.asarray([rsh, rsgrid[x], a, i0, ipv])
#     rms[x] = DE.objf(v, expdata, vth, 1, 1)
#
# plt.plot(rsgrid, rms, 'go')
# plt.grid
# plt.show()

# a
# rsh = 54.1134
# i0 = 3.209e-07
# ipv = 0.7607
# rs = 0.0363
# agrid = np.arange(1, 2, 0.01)
# temp = 33 + 275.15
# vth = constants.Boltzmann * temp / constants.elementary_charge
#
# rms = np.zeros(agrid.shape)
#
# for x in range(agrid.size):
#     v = np.asarray([rsh, rs, agrid[x], i0, ipv])
#     rms[x] = DE.objf(v, expdata, vth, 1, 1)
#
# plt.plot(agrid, rms, 'go')
# plt.grid()
# plt.ylim([0, 2.5])
# plt.xlim([1, 2])
# plt.show()

# STP6

# expdata = read_csv("data/STP6")
#
# Rs = 0.005739182892356806
# a = 1.171311536180859
# I0 = 7.549548114835498e-07
# Ipv = 7.444819124356852
# Rsh = np.arange(0.1, 250, 1)
# T = 55 + 275.15
# vth = constants.Boltzmann * T / constants.elementary_charge
# rms = np.zeros(Rsh.shape)
#
# for x in range(Rsh.size):
#     v = np.asarray([Rsh[x], Rs, a, I0, Ipv])
#     rms[x] = DE.objf(v, expdata, vth, 36, 1)
#
# min = np.min(rms)
# mask = np.array(rms) == min
# color = np.where(mask, 'red', 'blue')
#
# plt.scatter(Rsh, rms, color=color)
# plt.grid()
# plt.show()
# print(rms)