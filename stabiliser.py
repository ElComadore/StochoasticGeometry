from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mystic.solvers import diffev2
from mystic.monitors import VerboseMonitor
import mystic.constraints


def generate_wiener_diff(delta, n):
    w = list()
    for i in range(n):
        w.append(np.random.normal(0, np.sqrt(delta)))

    return w


def milstein(x, mu, sigma, deriv, delta, omega):
    return x + mu * delta + sigma * omega + deriv * (omega * omega - delta)


def euler(x, mu, sigma, delta, omega):
    return x + mu * delta + sigma * omega


def base_processes(delta, wiener, x_0, x_mu, x_sig, y_0, y_mu, y_sig):
    x = list()
    y = list()

    x.append(x_0)
    y.append(y_0)

    for i in range(len(wiener)):
        x_new = euler(x[-1], x_mu(x[-1]), x_sig(x[-1]), delta, wiener[i])
        y_new = euler(y[-1], y_mu(x[-1], y[-1]), y_sig(x[-1], y[-1]), delta, wiener[i])

        x.append(x_new)
        y.append(y_new)

    return x, y


def all_processes(surf, tol, delta, wiener, x_0, x_mu, x_sig, y_0, y_mu, y_sig):
    x = list()
    y = list()
    stabs = list()

    x.append(x_0)
    y.append(y_0)

    C = surf(x_0, y_0)

    for i in range(len(wiener)):
        x_new = euler(x[-1], x_mu(x[-1]), x_sig(x[-1]), delta, wiener[i])
        y_new = euler(y[-1], y_mu(x[-1], y[-1]), y_sig(x[-1], y[-1]), delta, wiener[i])

        if abs(surf(x_new, y_new) - C) > tol:
            stabs.append(i)
            print(i)
            x_new, y_new = stabilise([x_new, y_new], surf, C)

        x.append(x_new)
        y.append(y_new)
    return x, y, stabs


def stabilise(x_0, surf, level):
    def dist_min(x):
        return (x[0] - x_0[0]) ** 2 + (x[1] - x_0[1]) ** 2

    def constraint(x):
        return surf(x[0], x[1]) - level

    @mystic.penalty.lagrange_equality(constraint, k=1000)
    def penalty(x):
        return 0

    bounds = [(0, 10000)] * 2
    mon = VerboseMonitor(10)

    result = diffev2(dist_min, x0=x_0, penalty=penalty, bounds=bounds, npop=100, gtol=50, disp=False, full_output=True,
                     ftol=0.0000005)

    return result[0][0], result[0][1]


def x_drift(x):
    return x


def x_diff(x):
    return x


def y_drift(x, y):
    return 0


def y_diff(x, y):
    return -y


def surface(x, y):
    return x*y


if __name__ == '__main__':
    print("I'm in")

    np.random.seed(270300)
    # np.random.seed(160899)

    num = 1000000
    d = 0.000001
    step = 0.0000001
    t = 0.0001
    x_0 = 1
    y_0 = 1

    white_noise = generate_wiener_diff(d, num)

    base_X, base_Y = base_processes(d, white_noise, x_0, x_drift, x_diff, y_0, y_drift, y_diff)

    X_t, Y_t, dev = all_processes(surface, t, d, white_noise, x_0,  x_drift, x_diff, y_0, y_drift, y_diff)

    print("\nSimulation Complete")

    s_base = 0
    s_x = 0
    s_stab = 0

    for j in range(num):
        s_base += abs(surface(base_X[j], base_Y[j]) - surface(x_0, y_0)) ** 2
        s_x += abs(base_X[j] - X_t[j]) ** 2
        s_stab += abs(surface(X_t[j], Y_t[j]) - surface(x_0, y_0)) ** 2

    print('\n')
    print("Unstabilised level: " + str(surface(base_X[-1], base_Y[-1])))
    print("Stabilised level: " + str(surface(X_t[-1], Y_t[-1])))
    print("Number of stabilisers: " + str(len(dev)))
    print('\n')
    print("Square Error without stabiliser: " + str(s_base))
    print("Square Error with stabiliser: " + str(s_stab))
    print("Square Error in X_t: " + str(s_x))

    dev_diff = list()
    if len(dev) > 0:
        dev_diff.append(dev[0])
        for j in range(1, len(dev)):
            dev_diff.append(dev[j] - dev[j - 1])

    fig, ax = plt.subplots(3, 2)
    ax[0, 0].plot(range(num+1), base_X)
    ax[0, 1].plot(range(num+1), X_t)
    ax[1, 0].plot(range(num + 1), X_t)

    for d in dev:
        ax[1, 0].plot(d, X_t[d], color='r', marker="o", markersize=2)
    ax[1, 1].plot(X_t, Y_t)
    ax[2, 0].hist(dev_diff, len(dev)//3, rwidth=0.8)
    ax[2, 1].plot(range(num+1), np.subtract(base_X, X_t))

    plt.show()
