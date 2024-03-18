import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time


def generate_wiener_diff(delta, n):
    w = list()
    for i in range(n):
        w.append(np.random.normal(0, np.sqrt(delta)))

    return w


def jiggle(x, mu, sigma, delta):
    """
    Jiggle the stochastic process forward 1 delta time step using left hand approximations
    :param x: the current value of the process
    :param mu: the drift
    :param sigma: the diffusion
    :param delta: the length of the time step
    :return: the new position
    """

    return x + mu*delta + sigma*np.random.normal(0, np.sqrt(delta))


def milstein(x, mu, sigma, deriv, delta, omega):
    return x + mu*delta + sigma*omega + deriv*(omega*omega - delta)


def euler(x, mu, sigma, delta, omega):
    return x + mu*delta + sigma*omega


def stochastic_processes(surf, h, alpha, beta, x0, y0, wiener, x_mu, x_sigma, x_deriv, y_mu, y_sigma, y_deriv, delta, n):
    x = list()
    y = list()

    x.append(x0)
    y.append(y0)

    for i in range(n):
        # x_new = milstein(x[-1], x_mu(surf, x[-1], y[-1]), x_sigma(x[-1], y[-1]), x_deriv(x[-1], y[-1]), delta,
        # wiener[i])
        # y_new = milstein(y[-1], y_mu(x[-1], y[-1]), y_sigma(x[-1], y[-1]), y_deriv(x[-1], y[-1]), delta, wiener[i])

        x_new = euler(x[-1], x_mu(surf, x[-1], y[-1], h, alpha, beta), x_sigma(surf, x[-1], y[-1], h, beta), delta,
                      wiener[i])
        y_new = euler(y[-1], y_mu(surf, x[-1], y[-1], h, alpha, beta), y_sigma(surf, x[-1], y[-1], h, beta), delta,
                      wiener[i])
        x.append(x_new)
        y.append(y_new)

    return x, y


def x_drift_add(surf, x, y, h, alpha, beta):
    return -0.5*beta*beta*((surf(x, y + h) - 2*surf(x, y) + surf(x, y - h))/(h*h))*((surf(x + h/2, y) - surf(x - h/2, y))/h) \
           + alpha*(surf(x, y + h/2) - surf(x, y - h/2))/h


def x_drift_multi(surf, x, y, h, alpha, beta):
    return -0.5*beta*beta*((surf(x, y + h) - 2*surf(x, y) + surf(x, y - h))/(h*h))*((surf(x + h/2, y) - surf(x - h/2, y))/h) \
           - alpha*(((surf(x-h, y-h) + surf(x + h, y + h) - surf(x + h, y - h) - surf(x - h, y + h))*beta*beta*(surf(x, y + h/2) - surf(x, y - h/2)))/(4*h*h*h))


def x_diff(surf, x, y, h, beta):
    return beta*(surf(x, y + h/2) - surf(x, y - h/2))/h


def x_diff_deriv(x, y):
    return 0.5*x


def y_drift_add(surf, x, y, h, alpha, beta):
    return -0.5*beta*beta*((surf(x + h, y) - 2*surf(x, y) + surf(x - h, y))/(h*h))*((surf(x, y + h/2) - surf(x, y - h/2))/h) \
           - alpha*(surf(x + h/2, y) - surf(x - h/2, y))/h


def y_drift_multi(surf, x, y, h, alpha, beta):
    return -0.5*beta*beta*((surf(x + h, y) - 2*surf(x, y) + surf(x - h, y))/(h*h))*((surf(x, y + h/2) - surf(x, y - h/2))/h) \
           + (alpha - 1)*(surf(x-h, y-h) + surf(x + h, y + h) - surf(x + h, y - h) - surf(x - h, y + h))*beta*beta*(surf(x + h/2, y) - surf(x - h/2, y))/(4*h*h*h)


def y_diff(surf, x, y, h, beta):
    return -beta*(surf(x + h/2, y) - surf(x - h/2, y))/h


def y_diff_deriv(x, y):
    return -0.5*y


def surface(x, y):
    return y - np.exp(-x)


if __name__ == '__main__':

    np.random.seed(270300)
    num = 1000000
    d = 0.00000001
    step = 0.000000001
    x_0 = -1
    y_0 = 0
    a = -4
    b = 16
    white_noise = generate_wiener_diff(d, num)
    x_drift = x_drift_add
    y_drift = y_drift_add

    X_t, Y_t = stochastic_processes(surface, step, a, b, x_0, y_0, white_noise, x_drift, x_diff, x_diff_deriv, y_drift,
                                    y_diff, y_diff_deriv, d, num)

    fig, ax = plt.subplots(1, 1)
    ax.plot(X_t, Y_t)
    # plt.xticks(np.arrange(-1, 1.1, step=0.1))
    # plt.yticks(np.arrange(-1, 1.1, step=0.1))
    ax.axis('equal')

    s = 0

    for j in range(num):
        s += abs(surface(X_t[j], Y_t[j]) - surface(x_0, y_0))

    print(surface(x_0, y_0))
    print(s)

    plt.show()
