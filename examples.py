from objects import *
import scipy.constants as sc

# RTC FRANCE #######################
b = {'rp':  [2, 100],
     'rs':  [0, 1],
     'a':   [1, 2],
     'i0':  [1e-07, 1e-04],
     'ipv': [0, 10]
     }
exp = read_csv("data/RTC33D1000W.csv")
T = 33 + 275.15
algo = DE(b, exp, 1, 1, T, mutf=0.8, crossr=1)
algo.solve()
algo.plot_fit_hist()
algo.plot_result(print_params=True)

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
# T = 33 + 275.15
# algo = DE(b, [*read_csv("data/RTC33D1000W.csv")], 1, 1, T)
# algo.solve()
# algo.plot_fit_hist()
# algo.plot_result(print_params=True)

# Schutten solar STP 6 SINGLE DIODE #########################################
# b = {'rp':  [5, 20],
#      'rs':  [0, 0.1],
#      'a':   [1, 2],
#      'i0':  [1e-07, 2e-6],
#      'ipv': [0, 10]
#      }
#
# T = 55 + 275.15
# algo = DE(b, [*read_csv("data/STP6")], 36, 1, T)
# algo.solve()
# algo.plot_fit_hist()
# algo.plot_result(print_params=True)

# Schutten solar STP 6 DOUBLE DIODE #########################################
# b = {'rp':   [2, 15],
#      'rs':   [0, 1],
#      'a1':   [1, 2],
#      'a2':   [1, 2],
#      'i01':  [1e-07, 1e-04],
#      'i02':  [1e-07, 1e-04],
#      'ipv':  [0, 10]
#      }
#
# T = 55 + 275.15
# algo = DE(b, [*read_csv("data/STP6")], 36, 1, T)
# algo.solve()
# algo.plot_fit_hist()
# algo.plot_result(print_params=True)


# STM 6 Single Diode #######################
# b = {'rp':  [2, 30],
#      'rs':  [0, 1],
#      'a':   [1, 2],
#      'i0':  [1e-07, 1e-5],
#      'ipv': [0, 10]
#      }
#
# T = 51 + 275.15
# algo = DE(b, [*read_csv("data/STM6_4036")], 36, 1, T)
# algo.solve()
# algo.plot_fit_hist()
# algo.plot_result(print_params=True)
#
# STM6 Double Diode ##########################
# b = {'rp':  [2, 1000],
#      'rs':  [0, 1],
#      'a1':  [1, 2],
#      'a2':  [1, 2],
#      'i01': [0, 1e-4],
#      'i02': [0, 1e-4],
#      'ipv': [0, 10]
#      }
# T = 51 + 275.15
# algo = DE(b, [*read_csv("data/STM6_4036")], 36, 1, T, popsize=400)
# algo.solve()
# algo.plot_fit_hist()
# algo.plot_result(print_params=True)

# # Photowatt PWP SINGLE DIODE ##########################
# b = {'rp':     [2, 100],
#      'rs':     [0, 1],
#      'a':      [1, 2],
#      'i0':     [1e-7, 1e-5],
#      'ipv':    [0, 5]
#      }
# T = 45 + 275.15
# algo = DE(b, [*read_csv("data/PWP")], 36, 1, T)
# algo.solve()
# algo.plot_fit_hist()
# algo.plot_result(print_params=True)

# # Photowatt PWP DOUBLE DIODE ##########################
# b = {'rp':  [2, 100],
#      'rs':  [0, 1],
#      'a1':  [1, 2],
#      'a2':  [1, 2],
#      'i01': [1e-7, 1e-5],
#      'i02': [1e-7, 1e-5],
#      'ipv': [0, 5]
#      }
# T = 45 + 275.15
# algo = DE(b, [*read_csv("data/PWP")], 36, 1, T)
# algo.solve()
# algo.plot_fit_hist()
# algo.plot_result(print_params=True)

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

# ST40 Thin Film single
# b = {'rp':  [2, 50],
#      'rs':  [0, 0.1],
#      'a':   [1, 2],
#      'i0':  [1e-06, 1e-04],
#      'ipv': [0, 1]
#      }
#
# T = 25 + 275.15
# algo = DE(b, [*read_csv("/home/youssef/PycharmProjects/DEPV/data/st40/200w.csv")], 42, 1, T, maxiter=50)
# algo.solve()
# algo.plot_fit_hist()
# algo.plot_result(print_params=True)

# ST40 Thin Film double
#
# b = {'rp':  [2, 50],
#      'rs':  [0, 0.1],
#      'a1':  [1, 2],
#      'a2':  [1, 2],
#      'i02': [1e-07, 1e-04],
#      'i0':  [1e-07, 1e-04],
#      'ipv': [0, 1]
#      }
#
# T = 25 + 275.15
# algo = DE(b, [*read_csv("/home/youssef/PycharmProjects/DEPV/data/st40/200w.csv")], 42, 1, T)
# algo.solve()
# algo.plot_fit_hist()
# algo.plot_result(print_params=True)
