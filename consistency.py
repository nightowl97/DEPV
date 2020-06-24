from objects import DE, read_csv
import numpy as np
import matplotlib.pyplot as plt


b = {'rp':  [2, 100],
     'rs':  [0, 1],
     'a':   [1, 2],
     'i0':  [1e-07, 1e-04],
     'ipv': [0, 10]
     }

T = 45 + 275.15
history = np.full(30, None, dtype=np.float64)
rs_hist = np.full(30, None, dtype=np.float64)
a_hist = np.full(30, None, dtype=np.float64)
i0_hist = np.full(30, None, dtype=np.float64)


for i in range(30):
    algo = DE(b, [*read_csv("data/PWP")], 36, 1, T)
    algo.solve()
    history[i] = algo.final_res[1]
    rs_hist[i] = algo.final_res[0][1]
    a_hist[i] = algo.final_res[0][2]
    i0_hist[i] = algo.final_res[0][3]

plt.scatter(range(1, 31), history)
plt.xlabel("Essais")
plt.ylabel("Valeur fitness")
plt.grid()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

plt.scatter(range(1, 31), rs_hist)
plt.ylabel(r"$R_s$")
plt.ylim([0, 0.1])
plt.grid()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

plt.scatter(range(1, 31), a_hist)
plt.ylabel(r"$a$")
plt.ylim([1, 2])
plt.grid()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

plt.scatter(range(1, 31), i0_hist)
plt.ylabel(r"$I_0$")
plt.grid()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

print(np.mean(history))
print(np.mean(rs_hist))
print(np.mean(a_hist))
print(np.mean(i0_hist))

print("----")
print(np.std(history))
print(np.std(rs_hist))
print(np.std(a_hist))
print(np.std(i0_hist))
