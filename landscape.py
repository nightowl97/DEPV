import matplotlib.pyplot as plt
import numpy as np
from testing import i_from_v
from objects import DE, read_csv
import scipy.constants as constants
from matplotlib import cm

# Experimental data
expdata = read_csv("/home/youssef/PycharmProjects/DEPV/data/RTC33D1000W.csv")

rsh = 54.1134
i0 = 3.209e-07
ipv = 0.7607
rsgrid = np.arange(0.035, 0.037, 0.0001)
agrid = np.arange(1.47, 1.48, 0.001)
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

ax.scatter(RS, A, Z, cmap='viridis')
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