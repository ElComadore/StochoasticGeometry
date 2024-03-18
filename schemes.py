import numpy as np


def generate_wiener_diff(delta, n):
    w = list()
    for i in range(n):
        w.append(np.random.normal(0, np.sqrt(delta)))

    return w


def euler(x, mu, sigma, delta, omega):
    return x + mu*delta + sigma*omega


def milstein(x, mu, sigma, sigma_derivative, delta, omega):
    return x + mu*delta + sigma*omega + 0.5*(omega*omega - delta)*sigma_derivative


def runge_kutta(x, mu, sigma, sigma_support_diff, delta, omega):
    return x + mu*delta + sigma*omega + 0.5*(1/(np.sqrt(delta)))*(omega*omega - delta)*sigma_support_diff
