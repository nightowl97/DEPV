from objects import DE, read_csv
import numpy as np
import matplotlib.pyplot as plt

# Example data
b = {'rp':  [2, 100],
     'rs':  [0, 1],
     'a':   [1, 2],
     'i0':  [1e-07, 1e-04],
     'ipv': [0, 10]}

T = 33 + 275.15

iternum = 20
popsize = 20
F = 0.8
CR = 0.8

populations = np.full((iternum, popsize, 2), None, dtype=np.float64)
fitnesses = np.full((iternum, popsize), 100, dtype=np.float64)
fitness_hist = []
avg_fit_hist = []


def get_final_res(vec):
    de_obj = DE(b, [*read_csv("/home/youssef/PycharmProjects/DEPV/data/RTC33D1000W.csv")], 1, 1, T, popsize=50, maxiter=30,
                mutf=vec[0], crossr=vec[1])
    de_obj.solve()
    return de_obj.final_res[1]


# initial population
populations[0] = np.random.uniform([0.5, 0.5], [1, 1], size=(popsize, 2))
fitnesses[0] = [get_final_res(vec) for vec in populations[0]]


for gen in range(iternum):
    print(gen)

    fittest_index = np.argmin(fitnesses[gen])
    fittest = populations[gen, fittest_index]

    fitness_hist.append(fitnesses[gen, fittest_index])
    avg_fit_hist.append(np.average(fitnesses[gen]))

    # iterate over pop
    for i in range(popsize):
        # Generate donor vector from 2 random vectors and the fittest
        indices = [ind for ind in range(popsize) if ind != i and ind != fittest_index]
        i1, i2 = np.random.choice(indices, 2, replace=False)

        # mutation
        mutant = fittest + F * (populations[gen, i1] - populations[gen, i2])
        # crossover
        crosspoints = np.random.rand(2) <= CR
        if not any(crosspoints):
            crosspoints[np.random.randint(0, 2)] = True
        trial = np.where(crosspoints, mutant, populations[gen, i])

        # penalty
        if trial[0] <= 0.5 or trial[0] > 1:
            trial[0] = np.random.uniform(low=0.5, high=1)
        if trial[1] <= 0.5 or trial[1] > 1:
            trial[1] = np.random.uniform(low=0.5, high=1)

        # selection
        f = get_final_res(trial)

        if gen + 1 < iternum:
            if f < fitnesses[gen, i]:
                populations[gen + 1, i] = trial
                fitnesses[gen + 1, i] = f
            else:
                populations[gen + 1, i] = populations[gen, i]
                fitnesses[gen + 1, i] = fitnesses[gen, i]


# final res
fittest_index = np.argmin(fitnesses[-1])
result = populations[-1][fittest_index]
result_fitness = fitnesses[-1][fittest_index]

print(result)
print(result_fitness)

plt.plot(avg_fit_hist, 'b')
plt.xlabel("Generation")
plt.ylabel("RMSE")
plt.show()

# plt.plot(fitness_hist, 'b')
# plt.xlabel("Generation")
# plt.ylabel("RMSE")
# plt.show()
