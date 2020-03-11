from objects import *

# RTC FRANCE #######################
# b = {'rp':  [2, 100],
#      'rs':  [0, 1],
#      'a':   [1, 2],
#      'i0':  [1e-07, 1e-04],
#      'ipv': [0, 10]
#      }
#
# algo = DE(b, [*read_csv("data/RTC33D1000W.csv")], 1, 1)
# T = 33 + 275.15
# Vt = sc.Boltzmann * T / sc.elementary_charge  # Thermal voltage
# algo.solve(Vt)
# algo.plot_fit_hist()
# algo.plot_result(Vt)

# Schutten solar STP 6 #########################################
# b = {'rp':  [2, 300],
#      'rs':  [0, 1],
#      'a':   [1, 2],
#      'i0':  [1e-07, 1e-5],
#      'ipv': [0, 10]
#      }
#
# algo = DE(b, [*read_csv("data/STP6")], 36, 1)
# T = 55 + 275.15
# Vt = sc.Boltzmann * T / sc.elementary_charge  # Thermal voltage
# algo.solve(Vt)
# algo.plot_fit_hist()
# algo.plot_result(Vt)

# STM 6 Single Diode #######################
b = {'rp':  [2, 300],
     'rs':  [0, 1],
     'a':   [1, 2],
     'i0':  [1e-07, 1e-5],
     'ipv': [0, 10]
     }

algo = DE(b, [*read_csv("data/STM6_4036")], 36, 1)
T = 51 + 275.15
Vt = sc.Boltzmann * T / sc.elementary_charge  # Thermal voltage
algo.solve(Vt)
algo.plot_fit_hist()
algo.plot_result(Vt)

# STM6 Double Dide ##########################
b = {'rp':  [2, 300],
     'rs':  [0, 1],
     'a1':   [1, 2],
     'a2':   [1, 2],
     'i01': [1e-7, 1e-5],
     'i02': [1e-7, 1e-5],
     'ipv': [0, 10]
     }
algo = DE(b, [*read_csv("data/STM6_4036")], 36, 1)
T = 51 + 275.15
Vt = sc.Boltzmann * T / sc.elementary_charge  # Thermal voltage
algo.solve(Vt)
algo.plot_fit_hist()
algo.plot_result(Vt)
