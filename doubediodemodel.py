import numpy as np
import scipy.constants as sc
import pvlib
import decimal
import csv
import matplotlib.pyplot as plt
from scipy.special import lambertw

#%% CONTROL PARAMETERS & CONSTANTS
T = 55 + 273.15  # Temperature
Ns = 36  # Number of cells in series
Np = 1  # Number of cells in parallel
Vt = sc.Boltzmann * T / sc.elementary_charge  # Thermal voltage
GENMAX = 300  # Max number of generation
NP = 100  # Population size
F = 0.7  # Mutation factor (Values > 0.4 are recommended [1])
CR = 0.8  # Crossover rate (High values recommended [2])
D = 7  # 7-Dimensional search space

# Search space
rs_l, rs_h = 0, 10 / Ns  # Series resistance
a1_l, a1_h = 1, 2  # Ideality factor boundaries
a2_l, a2_h = 1, 2
rp_l, rp_h = 5. / Ns, 400. / Ns  # Shunt resistance
ipv_l, ipv_h = 0, 10  # Photo-current
i0_l, i0_h = 0, 1.3e-04  # Saturation current

fit_hist = []  # Store fitness history
avg_fit_hist = []

with open("data/STP6") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader)  # Ignore the header
    data = [(float(row[0]), float(row[1])) for row in csv_reader]
    voltages, currents = np.array([row[0] for row in data]), np.array([row[1] for row in data])
    N = len(voltages)
    xlim = max(voltages) + int(.2 * max(voltages))
    ylim = max(currents) + int(.2 * max(currents)) + .02


#%% PIVOT POINT SELECTION
p1, p2 = data[0], data[-1]  # Isc and Voc
# Maximum power point (MPP)
powers = [a * b for a, b in zip(voltages, currents)]
p3 = None
p = 0  # First MPP
p3_fit = []  # store fittest of each p3 run
fittest = []


def i_from_v(resistance_shunt, resistance_series, n1, n2, voltage, saturation_current1, saturation_current2, photocurrent):
    nNsVth1, nNsVth2 = n1, n2
    output_is_scalar = all(map(np.isscalar,
                               [resistance_shunt, resistance_series, nNsVth1, nNsVth2,
                                voltage, saturation_current1, saturation_current2, photocurrent]))

    # This transforms Gsh=1/Rsh, including ideal Rsh=np.inf into Gsh=0., which
    #  is generally more numerically stable
    conductance_shunt = 1. / resistance_shunt

    # Ensure that we are working with read-only views of numpy arrays
    # Turns Series into arrays so that we don't have to worry about
    #  multidimensional broadcasting failing
    Gsh, Rs, a1, a2, V, I01, I02, IL = \
        np.broadcast_arrays(conductance_shunt, resistance_series, nNsVth1, nNsVth2,
                            voltage, saturation_current1, saturation_current2, photocurrent)

    # Intitalize output I (V might not be float64)
    I = np.full_like(V, np.nan, dtype=np.float64)  # noqa: E741, N806

    # Determine indices where 0 < Rs requires implicit model solution
    idx_p = 0. < Rs

    # Determine indices where 0 = Rs allows explicit model solution
    idx_z = 0. == Rs

    # Explicit solutions where Rs=0
    if np.any(idx_z):
        I[idx_z] = IL[idx_z] - I01[idx_z] * np.expm1(V[idx_z] / a1[idx_z]) \
                             - I02[idx_z] * np.expm1(V[idx_z] / a2[idx_z]) - \
                   Gsh[idx_z] * V[idx_z]

    # Only compute using LambertW if there are cases with Rs>0
    # Does NOT handle possibility of overflow, github issue 298
    if np.any(idx_p):
        # LambertW arguments, cannot be float128, may overflow to np.inf
        argw1 = (Rs[idx_p] * (I01[idx_p] + I02[idx_p]) / (a1[idx_p] * (Gsh[idx_p] * Rs[idx_p] + 1))) * np.exp(
            (Rs[idx_p] * (IL[idx_p] + I01[idx_p] + I02[idx_p]) + V[idx_p]) / (a1[idx_p] * (Gsh[idx_p] * Rs[idx_p] + 1))
        )
        argw2 = (Rs[idx_p] * (I01[idx_p] + I02[idx_p]) / (a2[idx_p] * (Gsh[idx_p] * Rs[idx_p] + 1))) * np.exp(
            (Rs[idx_p] * (IL[idx_p] + I01[idx_p] + I02[idx_p]) + V[idx_p]) / (a2[idx_p] * (Gsh[idx_p] * Rs[idx_p] + 1))
        )

        # lambertw typically returns complex value with zero imaginary part
        # may overflow to np.inf
        lambertwterm1 = lambertw(argw1).real
        lambertwterm2 = lambertw(argw2).real

        # Eqn. 2 in Jain and Kapoor, 2004
        #  I = -V/(Rs + Rsh) - (a/Rs)*lambertwterm + Rsh*(IL + I0)/(Rs + Rsh)
        # Recast in terms of Gsh=1/Rsh for better numerical stability.
        I[idx_p] = (IL[idx_p] + I01[idx_p] + I02[idx_p] - (V[idx_p] * Gsh[idx_p])) / (Gsh[idx_p] * Rs[idx_p] + 1) \
                   - (a1[idx_p] / (2 * Rs[idx_p])) * lambertwterm1 - (a2[idx_p] / (2 * Rs[idx_p])) * lambertwterm2

    if output_is_scalar:
        return I.item()
    else:
        return I


# Evaluation function using the three pivot points
def evaluate(rp, rs, a1, a2, i01, i02, ipv):
    # J function calculation (RMSE)
    # Find the current using the Lambert W function [5]
    ical = i_from_v(rp * Ns / Np, rs * Ns / Np, a1 * Ns * Vt, a2 * Ns * Vt, voltages, i01 * Np, i02 * Np, ipv * Np)
    j = currents - ical
    # Fitness penalty (J = 100) for unphysical values of Ipv, I0 and Rp
    if i01 < 0 or i02 < 0 or rp < 0 or ipv < 0:
        return 100

    return np.sqrt(np.mean(j ** 2))  # RMSE


# Plot vector
def plot_vec(sol_vec):
    # Plots I-V curve from the two parameters a, Rs
    plt.plot(voltages, currents, 'go')
    v = np.linspace(0, xlim, 100)
    rp, rs, a1, a2, i01, i02, ipv = sol_vec
    ical = i_from_v(rp * Ns / Np, rs * Ns / Np, a1 * Ns * Vt, a2 * Ns * Vt, v, i01 * Np, i02 * Np, ipv * Np)
    plt.plot(v, ical)
    plt.xlabel("Voltage V(V)")
    plt.ylabel("Current I(A)")
    plt.grid()
    plt.axis([0, xlim, 0, ylim])
    plt.show()
    plt.figure()


while p <= 1:

    gen = 1
    p3 = data[powers.index(max(powers)) + p]  # P(-1), P(0) and P(1)
    POP = np.random.uniform([rp_l, rs_l, a1_l, a2_l, i0_l, i0_l, ipv_l],
                            [rp_h, rs_h, a1_h, a2_h, i0_h, i0_h, ipv_h], size=(NP, D))  # Population initialization

    while gen <= GENMAX:

        fitness = np.asarray([evaluate(*vec) for vec in POP])
        fittest_index = np.argmin(fitness)
        fittest = POP[fittest_index]
        fit_hist.append(fitness[fittest_index])
        avg_fit_hist.append(np.average(fitness))

        for i in range(NP):
            rp, rs, a1, a2, i01, i02, ipv = POP[i]
            # Generate donor vector from 2 random vectors and the fittest
            indices = [ind for ind in range(NP) if ind != i and ind != fittest_index]
            i1, i2 = np.random.choice(indices, 2, replace=False)
            # DE mutation
            mutant = fittest + F * (POP[i1] - POP[i2])
            # Crossover
            crosspoints = np.random.rand(D) <= CR  # List of booleans
            if not any(crosspoints):
                crosspoints[np.random.randint(0, D)] = True  # Force at least one crosspoint
            trial = np.where(crosspoints, mutant, POP[i])
            # Penalty
            # Shunt resistance
            if not rp_l <= trial[0]:
                trial[0] = rp_l + np.random.rand() * (rp_h - rp_l)
            if not rp_h >= trial[0]:
                trial[0] = rp_h - np.random.rand() * (rp_h - rp_l)
            # Series resistance
            if not rs_l <= trial[1]:
                trial[1] = rs_l + np.random.rand() * (rs_h - rs_l)
            if not rs_h >= trial[1]:
                trial[1] = rs_h - np.random.rand() * (rs_h - rs_l)
            # First ideality factor
            if not a1_l <= trial[2]:
                trial[2] = a1_l + np.random.rand() * (a1_h - a1_l)
            if not a1_h >= trial[2]:
                trial[2] = a1_h - np.random.rand() * (a1_h - a1_l)
            # Second ideality factor
            if not a2_l <= trial[3]:
                trial[3] = a2_l + np.random.rand() * (a2_h - a2_l)
            if not a2_h >= trial[3]:
                trial[3] = a2_h - np.random.rand() * (a2_h - a2_l)
            # Saturation currents
            if not i0_l <= trial[4]:
                trial[4] = i0_l + np.random.rand() * (i0_h - i0_l)
            if not i0_h >= trial[4]:
                trial[4] = i0_h - np.random.rand() * (i0_h - i0_l)
            if not i0_l <= trial[5]:
                trial[5] = i0_l + np.random.rand() * (i0_h - i0_l)
            if not i0_h >= trial[5]:
                trial[5] = i0_l - np.random.rand() * (i0_h - i0_l)
            # Photo-current
            if not ipv_l <= trial[6]:
                trial[6] = ipv_l + np.random.rand() * (ipv_h - ipv_l)
            if not ipv_h >= trial[6]:
                trial[6] = ipv_h - np.random.rand() * (ipv_h - ipv_l)

            # Selection
            f = evaluate(*trial)
            if f < fitness[i]:
                fitness[i] = f
                POP[i] = trial  # Switch
                # Movie plotting
                # plt.plot(voltages, currents, 'go')
                # v = np.linspace(0, 25, 100)
                # rp, rs, a, i0, ipv = get_5_from_2(a, Rs, p3)
                # a1, a2, rs, rp, ipv, i01, i02 = *trial
                # ical = pvlib.
                # plt.plot(v, ical)
                # plt.xlabel("Voltage V(V)")
                # plt.ylabel("Current I(A)")
                # plt.grid()
                # plt.axis([0, 25, 0, 2])
                # camera.snap()

                if f < fitness[fittest_index]:
                    fittest_index = i
                    fittest = trial

        gen += 1
    plot_vec(fittest)
    p3_fit.append((*fittest, p3))
    p += 1

    # animation = camera.animate()
    # animation.save("evol{}-{}.mp4".format(p, gen))

fittest = p3_fit[np.argmin([evaluate(*e[:-1]) for e in p3_fit])]

print("RESULTS:\n(Rp:{}, Rs:{}, a1:{}, a2:{}, I01:{}, I02:{}, Ipv:{})".format(*fittest[:-1]))
print("RMSE: {:.2E}".format(decimal.Decimal(evaluate(*fittest[:-1]))))

# DATA REPRESENTATION
plot_vec(fittest[:-1])

plt.plot(avg_fit_hist, 'b')
plt.grid()
plt.xlabel("Generation")
plt.ylabel("RMSE")
plt.xlim((0, GENMAX - 1))
plt.show()
exit(0)
