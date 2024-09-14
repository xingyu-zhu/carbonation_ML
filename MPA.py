from numpy import zeros, sort, argsort, sin, abs, argmin
from random import random
from math import gamma, pi
from copy import copy
from numpy.random import randn, random, randint

def Initialization(pop, dim, ub, lb):
    X = zeros(shape=[pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub

def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]

    return X

def CalculateFitness(X, function):
    pop = X.shape[0]
    fitness = zeros(shape=[pop, 1])
    for i in range(pop):
        fitness[i] = function(X[i, :])

    return fitness

def SortFitness(Fit):
    fitness = sort(a=Fit, axis=0)
    index = argsort(a=Fit, axis=0)

    return fitness, index

def SortPosition(X, index):
    X_new = zeros(shape=X.shape)
    for i in range(X.shape[0]):
        X_new[i, :] = X[index[i], :]

    return X_new

def Levy(d):
    beta = 3 / 2
    sigma = ((gamma(1+beta) * sin(pi * beta / 2)) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
             ** (1 / beta))
    u = randn(1, d) * sigma
    v = randn(1, d)
    step = u / abs(v) ** (1 / beta)
    L = 0.05 * step

    return L

def MPA(pop, dim, lb, ub, MaxIter, function):
    P = 0.5
    FADS = 0.2
    X, lb, ub = Initialization(pop, dim, ub, lb)
    fitness = CalculateFitness(X, function)
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex)
    BestScore = copy(fitness[0])
    BestPosition = zeros([1, dim])
    BestPosition[0, :] = copy(X[0, :])
    Curve = zeros([MaxIter, 1])
    X_new = copy(X)
    for t in range(MaxIter):
        print("第" + str(t) + "次迭代")
        RB = randn(pop, dim)
        L = Levy(dim)
        CF = (1 - t / MaxIter) ** (2 * t / MaxIter)
        for i in range(pop):
            for j in range(dim):
                if t < MaxIter / 3:
                    stepSize = RB[i, j] * (BestPosition[0, j] - RB[i, j] * X[i, j])
                    X_new[i, j] = X[i, j] + P * random() * stepSize
                if MaxIter / 3 <= t <= 2 * MaxIter / 3:
                    if i < pop / 2:
                        stepSize = L[0, j] * (BestPosition[0, j] - L[0, j] * X[i, j])
                        X_new[i, j] = X[i, j] + P * random() * stepSize
                    else:
                        stepSize = RB[i, j] * (RB[i, j] * BestPosition[0, j] - X[i, j])
                        X_new[i, j] = X[i, j] + P * CF * stepSize
                if t > 2 * MaxIter / 3:
                    stepSize = L[0, j] * (L[0, j] * BestPosition[0, j] - X[i, j])
                    X_new[i, j] = X[i, j] + P * CF * stepSize

        X_new = BorderCheck(X_new, ub, lb, pop, dim)
        fitnessNew = CalculateFitness(X_new, function)
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                fitness[i] = copy(fitnessNew[i])
                X[i, :] = copy(X_new[i, :])

        indexBest = argmin(fitness)
        if fitness[indexBest] < BestScore:
            BestScore = copy(fitness[indexBest])
            BestPosition[0, :] = copy(X[indexBest, :])

        for i in range(pop):
            for j in range(dim):
                if random() < FADS:
                    U = random() < FADS
                    X_new[i, j] = X[i, j] * CF * (lb[j, 0] + random() * (ub[j, 0] - lb[j, 0]) * U)
                else:
                    r = random()
                    step_size = (FADS * (1 - r) + r) * (X[randint(pop), j] - X[randint(pop), j])
                    X_new[i, j] = X[i, j] + step_size
        X_new = BorderCheck(X_new, ub, lb, pop, dim)
        fitnessNew = CalculateFitness(X_new, function)
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                fitness[i] = fitnessNew[i].item()
                X[i, :] = copy(X_new[i, :])

        indexBest = argmin(fitness)
        if fitness[indexBest] < BestScore:
            BestScore = copy(fitness[indexBest])
            BestPosition[0, :] = copy(X[indexBest, :])

        Curve[t] = BestScore

    return BestScore, BestPosition, Curve
