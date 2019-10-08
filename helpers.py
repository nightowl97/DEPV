import numpy as np


def get_mpp(data):
    # grabs the data-point with the highest power
    voltages, currents = [float(tup[0]) for tup in data], [float(tup[1]) for tup in data]
    powers = [a * b for a, b in zip(voltages, currents)]
    # print(voltages, currents, powers)
    return data[powers.index(max(powers)) + 1]


# def check_unphysical(pop, p1, p2, p3):
