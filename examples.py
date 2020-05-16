from objects import *
import scipy.constants as sc

# RTC FRANCE #######################
# b = {'rp':  [2, 100],
#      'rs':  [0, 1],
#      'a':   [1, 2],
#      'i0':  [1e-07, 1e-04],
#      'ipv': [0, 10]
#      }
#
# algo = DE(b, [*read_csv("/home/youssef/PycharmProjects/DEPV/data/RTC33D1000W.csv")], 1, 1)
# T = 33 + 275.15
# algo.solve(T)
# algo.plot_fit_hist()
# algo.plot_result(T)

# RTC FRANCE DOUBLE DIODE #######################
# b = {'rp':  [2, 100],
#      'rs':  [0, 1],
#      'a1':   [1, 2],
#      'a2':   [1, 2],
#      'i01':  [1e-07, 1e-04],
#      'i02':  [1e-07, 1e-04],
#      'ipv': [0, 10]
#      }
#
# algo = DE(b, [*read_csv("data/RTC33D1000W.csv")], 1, 1)
# T = 33 + 275.15
# algo.solve(T)
# algo.plot_fit_hist()
# algo.plot_result(T)

# Schutten solar STP 6 SINGLE DIODE #########################################
# b = {'rp':  [2, 15],
#      'rs':  [0, 1],
#      'a':   [1, 2],
#      'i0':  [1e-07, 1e-5],
#      'ipv': [0, 10]
#      }
#
# algo = DE(b, [*read_csv("data/STP6")], 36, 1)
# T = 55 + 275.15
# algo.solve(T)
# algo.plot_fit_hist()
# algo.plot_result(T)

# Schutten solar STP 6 DOUBLE DIODE #########################################
# b = {'rp':  [2, 30],
#      'rs':  [0, 1],
#      'a1':   [1, 2],
#      'a2':   [1, 2],
#      'i01':  [1e-07, 1e-04],
#      'i02':  [1e-07, 1e-04],
#      'ipv': [0, 10]
#      }
#
# algo = DE(b, [*read_csv("data/STP6")], 36, 1, maxiter=400)
# T = 55 + 275.15
# algo.solve(T)
# algo.plot_fit_hist()
# algo.plot_result(T)


# STM 6 Single Diode #######################
# b = {'rp':  [2, 300],
#      'rs':  [0, 1],
#      'a':   [1, 2],
#      'i0':  [1e-07, 1e-5],
#      'ipv': [0, 10]
#      }
#
# algo = DE(b, [*read_csv("data/STM6_4036")], 36, 1)
# T = 51 + 275.15
# algo.solve(T)
# algo.plot_fit_hist()
# algo.plot_result(T)
#
# STM6 Double Diode ##########################
# b = {'rp':  [2, 300],
#      'rs':  [0, 1],
#      'a1':  [1, 2],
#      'a2':  [1, 2],
#      'i01': [1e-7, 1e-5],
#      'i02': [1e-7, 1e-5],
#      'ipv': [0, 10]
#      }
# algo = DE(b, [*read_csv("data/STM6_4036")], 36, 1)
# T = 51 + 275.15
# algo.solve(T)
# algo.plot_fit_hist()
# algo.plot_result(T)

# # Photowatt PWP SINGLE DIODE ##########################
# b = {'rp':  [2, 100],
#      'rs':  [0, 1],
#      'a':  [1, 2],
#      'i0': [1e-7, 1e-5],
#      'ipv': [0, 5]
#      }
# algo = DE(b, [*read_csv("data/PWP")], 36, 1)
# T = 45 + 275.15
# algo.solve(T)
# algo.plot_fit_hist()
# algo.plot_result(T)

# # Photowatt PWP DOUBLE DIODE ##########################
# b = {'rp':  [2, 100],
#      'rs':  [0, 1],
#      'a1':  [1, 2],
#      'a2':  [1, 2],
#      'i01': [1e-7, 1e-5],
#      'i02': [1e-7, 1e-5],
#      'ipv': [0, 5]
#      }
# algo = DE(b, [*read_csv("data/PWP")], 36, 1)
# T = 45 + 275.15
# algo.solve(T)
# algo.plot_fit_hist()
# algo.plot_result(T)

# SiCell ############################33
# b = {'rp':  [2, 100],
#      'rs':  [0, 1],
#      'a':  [1, 2],
#      'i0': [1e-7, 1e-5],
#      'ipv': [0, 1]
#      }
# algo = DE(b, [*read_csv("data/sicell.txt")], 1, 1)
# T = 25 + 275.15
# algo.solve(T)
# algo.plot_fit_hist()
# algo.plot_result(T)

# SiCell Double Diode ############################33
# b = {'rp':  [2, 100],
#      'rs':  [0, 1],
#      'a1':  [1, 2],
#      'a2':  [1, 2],
#      'i01': [1e-7, 1e-5],
#      'i02': [1e-7, 1e-5],
#      'ipv': [0, 1]
#      }
# algo = DE(b, [*read_csv("data/sicell.txt")], 1, 1)
# T = 25 + 275.15
# algo.solve(T)
# algo.plot_fit_hist()
# algo.plot_result(T)


# BCZT ##############################
# b = {'rp':  [2, 100],
#      'rs':  [0, 1],
#      'a':  [1, 5],
#      'i0': [1e-7, 1e-5],
#      'ipv': [0, 1]
#      }
# algo = DE(b, [*read_csv("data/bczt200nm_-6_6")], 1, 1)
# T = 25 + 275.15
# algo.solve(T)
# algo.plot_fit_hist()
# algo.plot_result(T)
