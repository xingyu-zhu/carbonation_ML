from math import e

import numpy as np
from random import uniform
from numpy import sign


def sa_algorithm(func, x0, T_max, T_min, L, max_stay, lb, ub):
    x_current = x_best = x0[0]
    y_current = y_best = -func(x0)[0]
    stay = 0
    epoch = 0
    T = T_max
    generation_x_best = []
    generation_y_best = []

    while T >= T_min and stay <= max_stay:
        for j in range(L):
            x_new = []
            r = uniform(-1, 1)
            rand = uniform(0, 1)
            for i_th in range(len(x_current)):
                x_new.append(x_current[i_th] + sign(r) * T * (pow((1 + (1 / T)), abs(r)) - 1) * (ub[i_th] - lb[i_th]))
            x_new = np.clip(x_new, lb, ub)
            y_new = -func([x_new])[0]
            df = y_new - y_current
            if pow(e, -(df / T)) > rand:
                x_current = x_new
                y_current = y_new
                if df < 0:
                    x_best = x_new
                    y_best = y_new
        epoch += 1
        T = T_max * pow(e, -pow(e, -1) * epoch)
        generation_x_best.append(x_best)
        generation_y_best.append(y_best)
        if len(generation_y_best) > 1:
            if abs(generation_y_best[-1] - generation_y_best[-2]) < 1e-3:
                stay += 1
            else:
                stay = 0
        else:
            pass

    return generation_x_best, generation_y_best