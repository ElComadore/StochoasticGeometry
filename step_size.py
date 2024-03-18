import math

import numpy as np
import schemes
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.stats as sp


def one_step_prob_euler(start_point, drift, diffusion, step_size, white_noise, surface):
    sims = []
    scheme = schemes.euler
    for i in range(len(white_noise)):
        x = scheme(start_point[0], drift[0], diffusion[0], step_size, white_noise[i])
        y = scheme(start_point[1], drift[1], diffusion[1], step_size, white_noise[i])

        sims.append(surface(x, y))
    return sims


def one_step_prob_runge_kutta(start_point, drift, diffusion, diff_support, step_size, white_noise, surface):
    sims = []
    scheme = schemes.runge_kutta
    for i in range(len(white_noise)):
        x = scheme(start_point[0], drift[0], diffusion[0], diff_support[0], step_size, white_noise[i])
        y = scheme(start_point[1], drift[1], diffusion[1], diff_support[1], step_size, white_noise[i])

        sims.append(surface(x, y))
    return sims


def one_step_prob_milstein(start_point, drift, diffusion, diff_derivative, step_size, white_noise, surface):
    sims = []
    scheme = schemes.milstein
    for i in range(len(white_noise)):
        x = scheme(start_point[0], drift[0], diffusion[0], diff_derivative[0], step_size, white_noise[i])
        y = scheme(start_point[1], drift[1], diffusion[1], diff_derivative[1], step_size, white_noise[i])

        sims.append(surface(x, y))

    return sims


class Circle:
    def __init__(self):
        pass

    def mu_x(self, x, y):
        return -0.5 * x

    def mu_y(self, x, y):
        return -0.5 * y

    def diff_x(self, x, y):
        return -y

    def diff_y(self, x, y):
        return x

    def derivative(self, x, y):
        return [x, y]

    def support(self, x, y, step_size):
        supp_x = x + np.sqrt(step_size) * self.mu_x(x, y) + step_size * self.diff_x(x, y)
        supp_y = y + np.sqrt(step_size) * self.mu_y(x, y) + step_size * self.diff_y(x, y)
        supp_diff_x = self.diff_x(supp_x, supp_y) - self.diff_x(x, y)
        supp_diff_y = self.diff_y(supp_x, supp_y) - self.diff_y(x, y)
        return [supp_diff_x, supp_diff_y]

    def surface(self, x, y):
        return x ** 2 + y ** 2

    def rk_poly(self, z, step_size):
        x = 1
        y = 0

        supp_x = x + np.sqrt(step_size) * self.mu_x(x, y) + step_size * self.diff_x(x, y)
        supp_y = y + np.sqrt(step_size) * self.mu_y(x, y) + step_size * self.diff_y(x, y)

        d_1 = 0.25 * supp_y ** 2 + 0.25 * (supp_x - 1) ** 2
        d_2 = supp_x - 1
        d_3 = -0.5 * supp_y ** 2 + 1 - 0.5 * (supp_x - 1) ** 2
        d_4 = 0.5 * supp_y
        d_5 = -supp_y
        d_6 = 0
        d_7 = -1 + 0.25 * supp_y ** 2 + 0.25 * (supp_x - 1) ** 2
        d_8 = 0.25

        return (d_1 / step_size) * z ** 4 + (d_2 / np.sqrt(step_size)) * z ** 3 + \
               (d_3 + d_4 * np.sqrt(step_size) + d_5 / np.sqrt(step_size)) * z ** 2 + \
               (d_6 * step_size - d_2 * np.sqrt(step_size)) * z - d_5 * np.sqrt(step_size) + \
               d_7 * step_size - d_4 * np.power(step_size, 1.5) + d_8 * step_size ** 2

    def mil_poly(self, z, stepsize):
        d_1 = 0.25
        d_2 = 0
        d_3 = 2
        d_4 = -1
        d_5 = 0
        d_6 = -2
        d_7 = 1

        return d_1 * z ** 4 + d_2 * z ** 3 + (d_3 + d_4 * stepsize) * z ** 2 + d_5 * stepsize * z + \
               d_6 * stepsize + d_7 * stepsize ** 2


class Hyperbola:
    def __init__(self):
        pass

    def mu_x(self, x, y):
        return 0.5 * x

    def mu_y(self, x, y):
        return 0.5 * y

    def diff_x(self, x, y):
        return -y

    def diff_y(self, x, y):
        return -x

    def derivative(self, x, y):
        return [x, y]

    def support(self, x, y, step_size):
        supp_x = x + np.sqrt(step_size) * self.mu_x(x, y) + step_size * self.diff_x(x, y)
        supp_y = y + np.sqrt(step_size) * self.mu_y(x, y) + step_size * self.diff_y(x, y)
        supp_diff_x = self.diff_x(supp_x, supp_y) - self.diff_x(x, y)
        supp_diff_y = self.diff_y(supp_x, supp_y) - self.diff_y(x, y)
        return [supp_diff_x, supp_diff_y]

    def surface(self, x, y):
        return x ** 2 - y ** 2

    def euler_poly(self, z, step_size):
        d_1 = -1
        d_2 = 0
        d_3 = 1
        d_4 = 0.25

        return d_1 * z ** 2 + d_2 * step_size * z + d_3 * step_size + d_4 * step_size ** 2

    def rk_poly(self, z, step_size):
        x = 1
        y = 0

        supp_x = x + np.sqrt(step_size) * self.mu_x(x, y) + step_size * self.diff_x(x, y)
        supp_y = y + np.sqrt(step_size) * self.mu_y(x, y) + step_size * self.diff_y(x, y)

        d_1 = 0.25 * (supp_y ** 2) - 0.25 * ((supp_x - 1) ** 2)
        d_2 = 1 - supp_x
        d_3 = -0.5 * (supp_y ** 2) - (1 - 0.5 * (1 - supp_x) ** 2)
        d_4 = -0.5 * supp_y
        d_5 = -supp_y
        d_6 = 0
        d_7 = 1 + d_1
        d_8 = 0.25

        return (d_1 / step_size) * z ** 4 + (d_2 / np.sqrt(step_size)) * z ** 3 + \
               (d_3 + d_4 * np.sqrt(step_size) + d_5 / np.sqrt(step_size)) * z ** 2 + \
               (d_6 * step_size - d_2 * np.sqrt(step_size)) * z - d_5 * np.sqrt(step_size) + \
               d_7 * step_size - d_4 * np.power(step_size, 1.5) + d_8 * step_size ** 2

    def mil_poly(self, z, step_size):
        d_1 = 0.25
        d_2 = 0
        d_3 = 0
        d_4 = 0
        d_5 = 0
        d_6 = 0
        d_7 = 0
        return d_1 * z ** 4 + d_2 * z ** 3 + (d_3 + d_4 * step_size) * z ** 2 + d_5 * step_size * z + \
               d_6 * step_size + d_7 * step_size ** 2


def input_vs_actual():
    print("Running input v actual")
    np.random.seed(270301)

    epsilon = 0.05
    delts = 10
    n = 1000
    h_vec = np.linspace(0.01, 0.1, n)
    const = delts
    fail_vec_euler = np.zeros([2, n])
    fail_vec_rk = np.zeros([2, n])
    fail_vec_mil = np.zeros([2, n])
    trials = 1000

    start = [1, 0]
    circle = Circle()
    hyper = Hyperbola()

    probs_euler = np.zeros((2, n))
    probs_rk = np.zeros((2, n))
    probs_mil = np.zeros((2, n))

    loc = 0
    scale = np.sqrt(h_vec)

    first = True
    first_rk = True
    first_mil = True

    first_eul_hype = True
    first_rk_hype = True

    mu = [[circle.mu_x(start[0], start[1]), circle.mu_y(start[0], start[1])],
          [hyper.mu_x(start[0], start[1]), hyper.mu_y(start[0], start[1])]]
    sigma = [[circle.diff_x(start[0], start[1]), circle.diff_y(start[0], start[1])],
             [hyper.diff_x(start[0], start[1]), hyper.diff_y(start[0], start[1])]]

    sigma_derivative = [circle.derivative(start[0], start[1]), hyper.derivative(start[0], start[1])]

    for i in range(len(h_vec)):
        zero = np.sqrt(epsilon + h_vec[i] - 0.25 * h_vec[i] ** 2)

        probs_euler[0, i] = 1 - sp.norm.cdf(zero, loc=loc, scale=scale[i]) + sp.norm.cdf(-zero, loc=loc, scale=scale[i])

        test = np.sqrt(-epsilon + h_vec[i] - 0.25 * h_vec[i] ** 2)
        if not math.isnan(test):
            if first:
                print(h_vec[i])
                print(h_vec[i] - 0.25 * h_vec[i] ** 2)
                first = False
            probs_euler[0, i] += sp.norm.cdf(test, loc=loc, scale=scale[i]) - sp.norm.cdf(-test, loc=loc,
                                                                                          scale=scale[i])

        x = np.linspace(loc - 4 * scale[i], loc + 4 * scale[i], 1000)

        poly = list()

        for j in range(len(x)):
            poly.append(circle.rk_poly(x[j], h_vec[i]))

        sign = 0
        prob = 0
        mem = None

        for j in range(len(poly)):
            if sign == 0 and poly[j] - epsilon > 0:
                sign = 1
            elif sign == 0 and poly[j] - epsilon < 0:
                sign = -1
            elif sign == 1 and poly[j] - epsilon < 0:
                sign = -1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i])
                else:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i]) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])
            elif sign == -1 and poly[j] - epsilon > 0:
                mem = j
                sign = 1

        if poly[-1] > epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=loc, scale=scale[i]) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])

        mem = None
        sign = 0
        for j in range(len(poly)):
            if sign == 0 and poly[j] < -epsilon:
                sign = -1
            elif sign == 0 and poly[j] > -epsilon:
                sign = 1
            if sign == -1 and poly[j] > -epsilon:
                sign = 1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i])
                else:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i]) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])
            if sign == 1 and poly[j] < -epsilon:
                if first_rk:
                    print("Rk Crit", h_vec[i])
                    first_rk = False
                mem = j
                sign = -1
        if poly[-1] < -epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=loc, scale=scale[i]) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])

        probs_rk[0, i] = prob

        poly = list()

        for j in range(len(x)):
            poly.append(circle.mil_poly(x[j], h_vec[i]))

        sign = 0
        prob = 0
        mem = None

        for j in range(len(poly)):
            if sign == 0 and poly[j] - epsilon > 0:
                sign = 1
            elif sign == 0 and poly[j] - epsilon < 0:
                sign = -1
            elif sign == 1 and poly[j] - epsilon < 0:
                sign = -1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i])
                else:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i]) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])
            elif sign == -1 and poly[j] - epsilon > 0:
                mem = j
                sign = 1

        if poly[-1] > epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=loc, scale=scale[i]) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])

        mem = None
        sign = 0
        for j in range(len(poly)):
            if sign == 0 and poly[j] < -epsilon:
                sign = -1
            elif sign == 0 and poly[j] > -epsilon:
                sign = 1
            if sign == -1 and poly[j] > -epsilon:
                sign = 1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i])
                else:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i]) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])
            if sign == 1 and poly[j] < -epsilon:
                if first_mil:
                    print('Mil Crit:', h_vec[i])
                    first_mil = False
                mem = j
                sign = -1
        if poly[-1] < -epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=loc, scale=scale[i]) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])

        probs_mil[0, i] = prob

        poly = list()

        for j in range(len(x)):
            poly.append(hyper.euler_poly(x[j], h_vec[i]))

        sign = 0
        prob = 0
        mem = None

        for j in range(len(poly)):
            if sign == 0 and poly[j] - epsilon > 0:
                sign = 1
            elif sign == 0 and poly[j] - epsilon < 0:
                sign = -1
            elif sign == 1 and poly[j] - epsilon < 0:
                sign = -1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i])
                else:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i]) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])
            elif sign == -1 and poly[j] - epsilon > 0:
                if first_eul_hype:
                    print("Euler Crit Hyper:", h_vec[i])
                    first_eul_hype = False
                    first_rk_hype = True
                mem = j
                sign = 1

        if poly[-1] > epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=loc, scale=scale[i]) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])

        mem = None
        sign = 0
        for j in range(len(poly)):
            if sign == 0 and poly[j] < -epsilon:
                sign = -1
            elif sign == 0 and poly[j] > -epsilon:
                sign = 1
            if sign == -1 and poly[j] > -epsilon:
                sign = 1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i])
                else:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i]) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])
            if sign == 1 and poly[j] < -epsilon:
                mem = j
                sign = -1
        if poly[-1] < -epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=loc, scale=scale[i]) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])

        probs_euler[1, i] = prob

        poly = list()

        for j in range(len(x)):
            poly.append(hyper.rk_poly(x[j], h_vec[i]))

        sign = 0
        prob = 0
        mem = None

        for j in range(len(poly)):
            if sign == 0 and poly[j] - epsilon > 0:
                sign = 1
            elif sign == 0 and poly[j] - epsilon < 0:
                sign = -1
            elif sign == 1 and poly[j] - epsilon < 0:
                sign = -1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i])
                else:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i]) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])
            elif sign == -1 and poly[j] - epsilon > 0:
                mem = j
                sign = 1

        if poly[-1] > epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=loc, scale=scale[i]) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])

        mem = None
        sign = 0
        for j in range(len(poly)):
            if sign == 0 and poly[j] < -epsilon:
                sign = -1
            elif sign == 0 and poly[j] > -epsilon:
                sign = 1
            if sign == -1 and poly[j] > -epsilon:
                sign = 1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i])
                else:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i]) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])
            if sign == 1 and poly[j] < -epsilon:
                if first_rk_hype:
                    print("Rk Crit Hyper:", h_vec[i])
                    first_rk_hype = False
                mem = j
                sign = -1
        if poly[-1] < -epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=loc, scale=scale[i]) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])

        probs_rk[1, i] = prob

        poly = list()

        for j in range(len(x)):
            poly.append(hyper.mil_poly(x[j], h_vec[i]))

        sign = 0
        prob = 0
        mem = None

        for j in range(len(poly)):
            if sign == 0 and poly[j] - epsilon > 0:
                sign = 1
            elif sign == 0 and poly[j] - epsilon < 0:
                sign = -1
            elif sign == 1 and poly[j] - epsilon < 0:
                sign = -1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i])
                else:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i]) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])
            elif sign == -1 and poly[j] - epsilon > 0:
                mem = j
                sign = 1

        if poly[-1] > epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=loc, scale=scale[i]) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])

        mem = None
        sign = 0
        for j in range(len(poly)):
            if sign == 0 and poly[j] < -epsilon:
                sign = -1
            elif sign == 0 and poly[j] > -epsilon:
                sign = 1
            if sign == -1 and poly[j] > -epsilon:
                sign = 1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i])
                else:
                    prob += sp.norm.cdf(0.5 * (x[j] + x[j - 1]), loc=loc, scale=scale[i]) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])
            if sign == 1 and poly[j] < -epsilon:
                mem = j
                sign = -1
        if poly[-1] < -epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=loc, scale=scale[i]) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=loc, scale=scale[i])

        probs_mil[1, i] = prob

        sigma_support = [circle.support(start[0], start[1], h_vec[i]), hyper.support(start[0], start[1], h_vec[i])]

        sim_euler = dict()
        sim_rk = dict()
        sim_mil = dict()

        noise = schemes.generate_wiener_diff(h_vec[i], trials)
        sim_euler["circle"] = one_step_prob_euler(start, mu[0], sigma[0], h_vec[i], noise, circle.surface)
        sim_rk["circle"] = one_step_prob_runge_kutta(start, mu[0], sigma[0], sigma_support[0],
                                                     h_vec[i], noise, circle.surface)
        sim_mil["circle"] = one_step_prob_milstein(start, mu[0], sigma[0], sigma_derivative[0],
                                                   h_vec[i], noise, circle.surface)

        sim_euler["hyper"] = one_step_prob_euler(start, mu[1], sigma[1], h_vec[i], noise, hyper.surface)
        sim_rk["hyper"] = one_step_prob_runge_kutta(start, mu[1], sigma[1], sigma_support[1],
                                                    h_vec[i], noise, hyper.surface)
        sim_mil["hyper"] = one_step_prob_milstein(start, mu[1], sigma[1], sigma_derivative[1],
                                                  h_vec[i], noise, hyper.surface)

        level = [circle.surface(start[0], start[1]), hyper.surface(start[0], start[1])]
        for j in range(len(noise)):
            if abs(sim_euler["circle"][j] - level[0]) > epsilon:
                fail_vec_euler[0][i] += 1
            if abs(sim_rk["circle"][j] - level[0]) > epsilon:
                fail_vec_rk[0][i] += 1
            if abs(sim_mil["circle"][j] - level[0]) > epsilon:
                fail_vec_mil[0][i] += 1
            if abs(sim_euler["hyper"][j] - level[1]) > epsilon:
                fail_vec_euler[1][i] += 1
            if abs(sim_rk["hyper"][j] - level[1]) > epsilon:
                fail_vec_rk[1][i] += 1
            if abs(sim_mil["hyper"][j] - level[1]) > epsilon:
                fail_vec_mil[1][i] += 1

    fail_est_euler = np.zeros([2, n])
    fail_est_rk = np.zeros([2, n])
    fail_est_mil = np.zeros([2, n])

    fail_est_euler[0] = np.divide(fail_vec_euler[0], trials)
    fail_est_euler[1] = np.divide(fail_vec_euler[1], trials)

    fail_est_rk[0] = np.divide(fail_vec_rk[0], trials)
    fail_est_rk[1] = np.divide(fail_vec_rk[1], trials)

    fail_est_mil[0] = np.divide(fail_vec_mil[0], trials)
    fail_est_mil[1] = np.divide(fail_vec_mil[1], trials)

    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    # fig, ax = plt.subplots(1, 3)
    # ax[0].plot(h_vec, fail_est_euler)
    # ax[1].plot(h_vec, fail_est_rk, color='r')
    # ax[2].plot(h_vec, fail_est_mil, color='orange')

    prob_est = np.multiply(h_vec, 0.1 * np.sqrt(2))

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(h_vec, fail_est_euler[0], label="Euler 0.5", color="royalblue")
    ax[0].plot(h_vec, fail_est_rk[0], label="Runge-Kutta 1.0", color="orange")
    ax[0].plot(h_vec, fail_est_mil[0], label="Milstein", color="red")
    ax[0].plot(h_vec, probs_euler[0], label="Analytic Euler", color="black", linestyle='--')
    ax[0].plot(h_vec, probs_rk[0], label="Analytic RK", color="black", linestyle='-.')
    ax[0].plot(h_vec, probs_mil[0], label="Analytic Mil", color="black", linestyle=':')
    ax[0].set_xlabel(r'Step-Size, \textit{h}', fontsize=22)
    ax[0].set_ylabel("Probability of Error", fontsize=22)
    # ax[0].xticks(np.linspace(0, 0.1, 11))
    # ax[0].yticks(np.linspace(0, 0.9, 11))
    ax[0].legend(fontsize=14)

    ax[1].plot(h_vec, fail_est_euler[1], label="Euler 0.5", color="royalblue")
    ax[1].plot(h_vec, fail_est_rk[1], label="Runge-Kutta 1.0", color="orange")
    ax[1].plot(h_vec, fail_est_mil[1], label="Milstein", color="red")
    ax[1].plot(h_vec, probs_euler[1], label="Analytic Euler", color="black", linestyle='--')
    ax[1].plot(h_vec, probs_rk[1], label="Analytic RK", color="black", linestyle='-.')
    ax[1].plot(h_vec, probs_mil[1], label="Analytic Mil", color="black", linestyle=':')
    ax[1].set_xlabel(r'Step-Size, \textit{h}', fontsize=22)
    ax[1].set_ylabel("Probability of Error", fontsize=22)
    # ax[1].xticks(np.linspace(0, 0.1, 11))
    # ax[1].yticks(np.linspace(0, 0.9, 11))
    ax[1].legend(fontsize=14)

    plt.suptitle("One-Step Error Probability for a Brownian Motion on a Circle and on a Hyperbola", fontsize=28)
    plt.show()


def error_hist():
    np.random.seed(270300)
    epsilon = 0.05
    delta = 1
    const = 10
    fails_euler = 0
    fails_rk = 0
    fails_mil = 0
    trials = 10000
    h = 0.01

    start = [1, 0]
    mu = [mu_x(start[0], start[1]), mu_y(start[0], start[1])]
    sigma = [diff_x(start[0], start[1]), diff_y(start[0], start[1])]
    sigma_support = support(start[0], start[1], h)
    sigma_derivative = derivative(start[0], start[1])

    noise = schemes.generate_wiener_diff(h, trials)
    sim_euler = one_step_prob_euler(start, mu, sigma, h, noise)
    sim_rk = one_step_prob_runge_kutta(start, mu, sigma, sigma_support, h, noise)
    sim_mil = one_step_prob_milstein(start, mu, sigma, sigma_derivative, h, noise)

    level = surface(start[0], start[1])
    for i in range(len(noise)):
        if abs(sim_euler[i] - level) > epsilon:
            fails_euler += 1
        if abs(sim_rk[i] - level) > epsilon:
            fails_rk += 1
        if abs(sim_mil[i] - level) > epsilon:
            fails_mil += 1

    fail_euler_est = np.divide(fails_euler, trials)
    fail_rk_est = np.divide(fails_rk, trials)
    fail_mil_est = np.divide(fails_mil, trials)
    print("Delta =", delta)
    print("Error Euler =", fail_euler_est)
    print("Error Runge-Kutta=", fail_rk_est)
    print("Error Milstein=", fail_mil_est)

    mean_sim = np.mean(sim_euler)
    mean_sim_2 = np.mean(sim_rk)
    var_sim = np.var(sim_euler)
    var_sim_2 = np.var(sim_rk)
    mean_sim_3 = np.mean(sim_mil)
    var_sim_3 = np.var(sim_mil)

    print("Euler Mean; ", mean_sim)
    print("Euler Var; ", var_sim)
    print("RK Mean; ", mean_sim_2)
    print("RK Var; ", var_sim_2)
    print("Milstein Mean; ", mean_sim_3)
    print("Milstein Var; ", var_sim_3)

    fig, ax = plt.subplots(1, 3)
    counts, bins = np.histogram(sim_euler, 50)
    ax[0].hist(bins[:-1], bins, weights=counts)
    counts, bins = np.histogram(sim_rk, 50)
    ax[1].hist(bins[:-1], bins, weights=counts, color='r')
    counts, bins = np.histogram(sim_mil, 50)
    ax[2].hist(bins[:-1], bins, weights=counts, color='orange')
    plt.show()


def tolerance_regression():
    np.random.seed(270300)

    l = 200

    epsilon = np.linspace(0.01, 0.25, l)
    h_critical = np.zeros([3, l])

    for e in range(len(epsilon)):
        n = 100
        h_vec = np.linspace(0.25 * epsilon[e], 10 * epsilon[e], n)

        fail_vec_euler = np.zeros([2, n])
        fail_vec_rk = np.zeros([2, n])
        fail_vec_mil = np.zeros([2, n])
        trials = 500

        start = [1, 0]
        circle = Circle()
        hyper = Hyperbola()

        mu = [[circle.mu_x(start[0], start[1]), circle.mu_y(start[0], start[1])],
              [hyper.mu_x(start[0], start[1]), hyper.mu_y(start[0], start[1])]]
        sigma = [[circle.diff_x(start[0], start[1]), circle.diff_y(start[0], start[1])],
                 [hyper.diff_x(start[0], start[1]), hyper.diff_y(start[0], start[1])]]

        sigma_derivative = [circle.derivative(start[0], start[1]), hyper.derivative(start[0], start[1])]

        for i in range(len(h_vec)):
            sigma_support = [circle.support(start[0], start[1], h_vec[i]), hyper.support(start[0], start[1], h_vec[i])]

            sim_euler = dict()
            sim_rk = dict()
            sim_mil = dict()

            noise = schemes.generate_wiener_diff(h_vec[i], trials)
            """
            sim_euler["circle"] = one_step_prob_euler(start, mu[0], sigma[0], h_vec[i], noise, circle.surface)
            sim_rk["circle"] = one_step_prob_runge_kutta(start, mu[0], sigma[0], sigma_support[0],
                                                         h_vec[i], noise, circle.surface)
            sim_mil["circle"] = one_step_prob_milstein(start, mu[0], sigma[0], sigma_derivative[0],
                                                       h_vec[i], noise, circle.surface)
            """

            sim_euler["hyper"] = one_step_prob_euler(start, mu[1], sigma[1], h_vec[i], noise, circle.surface)
            sim_rk["hyper"] = one_step_prob_runge_kutta(start, mu[1], sigma[1], sigma_support[1],
                                                        h_vec[i], noise, circle.surface)
            sim_mil["hyper"] = one_step_prob_milstein(start, mu[1], sigma[1], sigma_derivative[1],
                                                      h_vec[i], noise, circle.surface)

            level = [circle.surface(start[0], start[1]), hyper.surface(start[0], start[1])]
            for j in range(len(noise)):
                """
                if abs(sim_euler["circle"][j] - level[0]) > epsilon[e]:
                    fail_vec_euler[0][i] += 1
                if abs(sim_rk["circle"][j] - level[0]) > epsilon[e]:
                    fail_vec_rk[0][i] += 1
                if abs(sim_mil["circle"][j] - level[0]) > epsilon[e]:
                    fail_vec_mil[0][i] += 1
                """
                if abs(sim_euler["hyper"][j] - level[1]) > epsilon[e]:
                    fail_vec_euler[1][i] += 1
                if abs(sim_rk["hyper"][j] - level[1]) > epsilon[e]:
                    fail_vec_rk[1][i] += 1
                if abs(sim_mil["hyper"][j] - level[1]) > epsilon[e]:
                    fail_vec_mil[1][i] += 1

        fail_est_euler = np.zeros([2, n])
        fail_est_rk = np.zeros([2, n])
        fail_est_mil = np.zeros([2, n])

        fail_est_euler[0] = np.divide(fail_vec_euler[0], trials)
        fail_est_euler[1] = np.divide(fail_vec_euler[1], trials)

        fail_est_rk[0] = np.divide(fail_vec_rk[0], trials)
        fail_est_rk[1] = np.divide(fail_vec_rk[1], trials)

        fail_est_mil[0] = np.divide(fail_vec_mil[0], trials)
        fail_est_mil[1] = np.divide(fail_vec_mil[1], trials)

        """
        fail_diff_euler = fail_est_euler[0][1:] - fail_est_euler[0][:-1]
        fail_diff_rk = fail_est_rk[0][1:] - fail_est_rk[0][:-1]
        fail_diff_mil = fail_est_mil[0][1:] - fail_est_mil[0][:-1]
        
        m = max(fail_diff_euler)
        ind = np.where(fail_diff_euler == m)[0][0]
        h_critical[0][e] = h_vec[ind]

        m = max(fail_diff_rk)
        ind = np.where(fail_diff_rk == m)[0][0]
        h_critical[1][e] = h_vec[ind]

        m = max(fail_diff_mil)
        ind = np.where(fail_diff_mil == m)[0][0]
        h_critical[2][e] = h_vec[ind]
        """

        i = 0
        while i < n:
            if fail_est_euler[1][i] > 0.999:
                h_critical[0][e] = h_vec[i]
                break
            i += 1

        i = 0
        while i < n:
            if fail_est_rk[1][i] > 0.999:
                h_critical[1][e] = h_vec[i]
                break
            i += 1

    reg_euler = sp.linregress(epsilon, h_critical[0])
    reg_rk = sp.linregress(epsilon, h_critical[1])
    # reg_mil = sp.linregress((epsilon, h_critical[2]))

    print("Euler Slope:", reg_euler.slope)
    print("Runge-Kutta Slope:", reg_rk.slope)
    # print("Milstein Slope:", reg_mil.slope)

    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(epsilon, h_critical[0], label="Euler 0.5", color="blue")
    ax[0].set_xlabel(r"Tolerance, $\varepsilon$")
    ax[0].set_ylabel(r"Critical step-size estimate, \textit{h}")
    ax[0].set_title(r"Euler Scheme")

    ax[1].plot(epsilon, h_critical[1], label="Runge-Kutta 1.0", color="orange")
    ax[1].set_xlabel(r"Tolerance, $\varepsilon$")
    ax[1].set_ylabel(r"Critical step-size estimate, \textit{h}")
    ax[1].set_title(r"Runge-Kutta Scheme")

    """
    ax[2].plot(epsilon, h_critical[2], label="Milstein", color="red")
    ax[2].set_xlabel(r"Tolerance, $\varepsilon$")
    ax[2].set_ylabel(r"Critical step-size estimate, \textit{h}")
    ax[2].set_title(r"Milstein Scheme")
    ax[2].legend()
    """

    plt.show()


def poly_zeros():
    np.random.seed(270300)

    n = 100
    epsilon = 0.05
    h = 0.05
    h_vec = np.linspace(0.01, 0.1, n)
    probs = np.zeros((n, 1))

    first = True
    circle = Circle()

    """
        for i in range(len(h)):

        zero = np.sqrt(epsilon + h[i] - 0.25 * h[i] ** 2)

        probs[i] = 1 - sp.norm.cdf(zero, loc=mu, scale=sigma[i]) + sp.norm.cdf(-zero, loc=mu, scale=sigma[i])

        test = np.sqrt(-epsilon + h[i] - 0.25 * h[i] ** 2)
        if not math.isnan(test):
            if first:
                print(h[i])
                print(h[i] - 0.25 * h[i] ** 2)
                first = False
            probs[i] += sp.norm.cdf(test, loc=mu, scale=sigma[i]) - sp.norm.cdf(-test, loc=mu, scale=sigma[i])
    """

    for j in range(len(h_vec)):
        mu = 0
        sigma = np.sqrt(h_vec[j])
        poly = list()

        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

        for i in range(len(x)):
            poly.append(circle.mil_poly(x[i], h_vec[j]))

        sign = 0
        prob = 0

        mem = None

        for i in range(len(poly)):
            if sign == 0 and poly[i] - epsilon > 0:
                sign = 1
            elif sign == 0 and poly[i] - epsilon < 0:
                sign = -1
            elif sign == 1 and poly[i] - epsilon < 0:
                sign = -1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[i] + x[i - 1]), loc=mu, scale=sigma)
                else:
                    prob += sp.norm.cdf(0.5 * (x[i] + x[i - 1]), loc=mu, scale=sigma) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=mu, scale=sigma)
            elif sign == -1 and poly[i] - epsilon > 0:
                mem = i
                sign = 1

        if poly[-1] > epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=mu, scale=sigma) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=mu, scale=sigma)

        mem = None
        sign = 0
        for i in range(len(poly)):
            if sign == 0 and poly[i] < -epsilon:
                sign = -1
            elif sign == 0 and poly[i] > -epsilon:
                sign = 1
            if sign == -1 and poly[i] > -epsilon:
                sign = 1
                if mem is None:
                    prob += sp.norm.cdf(0.5 * (x[i] + x[i - 1]), loc=mu, scale=sigma)
                else:
                    prob += sp.norm.cdf(0.5 * (x[i] + x[i - 1]), loc=mu, scale=sigma) - \
                            sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=mu, scale=sigma)
            if sign == 1 and poly[i] < -epsilon:
                mem = i
                sign = -1
        if poly[-1] < -epsilon and mem is not None:
            prob += sp.norm.cdf(x[-1], loc=mu, scale=sigma) - \
                    sp.norm.cdf(0.5 * (x[mem] + x[mem - 1]), loc=mu, scale=sigma)

        probs[j] = prob

    plt.plot(h_vec, probs)
    # plt.plot(x, sp.norm.pdf(x, loc=mu, scale=sigma))
    # plt.plot(x, poly)
    # plt.hlines(0.05, mu-4*sigma, mu+4*sigma)
    # plt.hlines(-0.05, mu-4*sigma, mu+4*sigma)

    plt.show()


def intro_errors():
    np.random.seed(270300)

    m = 1000
    theta = np.linspace(0, 2 * np.pi, m)
    t = np.linspace(1, 2, m)
    ana_circle = np.zeros((2, m))
    ana_hyper = np.zeros((2, m))

    for i in range(m):
        ana_circle[0][i] = np.cos(theta[i])
        ana_circle[1][i] = np.sin(theta[i])

        ana_hyper[0][i] = np.sqrt(t[i] ** 2 - 1)
        ana_hyper[1][i] = -np.sqrt(t[i] ** 2 - 1)

    h = 0.000001
    n = int(10 / h)
    white_noise = schemes.generate_wiener_diff(h, n)

    h_vec = np.zeros(n + 1)
    h_vec[0] = 0
    for i in range(n):
        h_vec[i + 1] = h_vec[i] + h

    func = Circle()
    start = [1, 0]
    scheme = schemes.euler

    sim_euler = np.zeros((2, n + 1))
    sim_euler[0][0] = start[0]
    sim_euler[1][0] = start[1]

    sim_rk = np.zeros((2, n + 1))
    sim_rk[0][0] = start[0]
    sim_rk[1][1] = start[1]

    sim_mil = np.zeros((2, n + 1))
    sim_mil[0][0] = start[0]
    sim_mil[1][0] = start[1]

    error_index = list()

    for i in range(len(white_noise)):
        x = sim_euler[0][i]
        y = sim_euler[1][i]

        mu_x = func.mu_x(x, y)
        mu_y = func.mu_y(x, y)
        sigma_x = func.diff_x(x, y)
        sigma_y = func.diff_y(x, y)

        x_new = scheme(x, mu_x, sigma_x, h, white_noise[i])
        y_new = scheme(y, mu_y, sigma_y, h, white_noise[i])

        sim_euler[0][i + 1] = x_new
        sim_euler[1][i + 1] = y_new

        if abs(func.surface(x_new, y_new) - func.surface(start[0], start[1])) > 0.01:
            error_index.append(i)
        """
        x = sim_rk[0][i]
        y = sim_rk[1][i]

        mu_x = func.mu_x(x, y)
        mu_y = func.mu_y(x, y)
        sigma_x = func.diff_x(x, y)
        sigma_y = func.diff_y(x, y)
        supp = func.support(x, y, h)

        x_new = schemes.runge_kutta(x, mu_x, sigma_x, supp[0], h, white_noise[i])
        y_new = schemes.runge_kutta(y, mu_y, sigma_y, supp[1], h, white_noise[i])

        sim_rk[0][i + 1] = x_new
        sim_rk[1][i + 1] = y_new

        x = sim_mil[0][i]
        y = sim_mil[1][i]

        mu_x = func.mu_x(x, y)
        mu_y = func.mu_y(x, y)
        sigma_x = func.diff_x(x, y)
        sigma_y = func.diff_y(x, y)
        derivative = func.derivative(x, y)

        x_new = schemes.milstein(x, mu_x, sigma_x, derivative[0], h, white_noise[i])
        y_new = schemes.milstein(y, mu_y, sigma_y, derivative[1], h, white_noise[i])

        sim_mil[0][i + 1] = x_new
        sim_mil[1][i + 1] = y_new
        
    error_euler = 0
    error_rk = 0
    error_mil = 0
    level = func.surface(start[0], start[1])

    for i in range(len(sim_euler[0])):
        error_euler += abs(func.surface(sim_euler[0][i], sim_euler[1][i]) - level)
        error_rk += abs(func.surface(sim_rk[0][i], sim_rk[1][i]) - level)
        error_mil += abs(func.surface(sim_mil[0][i], sim_mil[1][i]) - level)

    print("Euler Error:", error_euler)
    print("RK Error:", error_rk)
    print("Mil Error:", error_mil)
    """

    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    print("Calcing deviates:", len(error_index))
    print("Fail Rate:", len(error_index) / n)
    deviates = np.zeros(len(error_index))
    for i in range(len(error_index)):
        deviates[i] = h_vec[error_index[i]]


    """
    fig, ax = plt.subplots(1, 3)
    
    ax[0].plot(sim_euler[0], sim_euler[1], label="Euler Method")
    # ax[0].plot(ana_circle[0], ana_circle[1], label="Unit Circle", color="black", linestyle="--")
    ax[0].plot(t, ana_hyper[0], label="Hyperbola", color="black", linestyle="--")
    ax[0].plot(t, ana_hyper[1], color="black", linestyle="--")
    ax[0].set_xlabel("x", fontsize=22)
    ax[0].set_ylabel("y", fontsize=22)
    ax[0].legend(fontsize=14)
    ax[0].set_aspect('equal', 'box')

    ax[1].plot(sim_rk[0], sim_rk[1], label="Runge-Kutta", color='orange')
    # ax[1].plot(ana_circle[0], ana_circle[1], label="Unit Circle", color="black", linestyle="--")
    ax[1].plot(t, ana_hyper[0], label="Hyperbola", color="black", linestyle="--")
    ax[1].plot(t, ana_hyper[1], color="black", linestyle="--")
    ax[1].set_xlabel("x", fontsize=22)
    ax[1].set_ylabel("y", fontsize=22)
    ax[1].legend(fontsize=14)
    ax[1].set_aspect('equal', 'box')
    ax[1].set_title("Three Simulations of the Brownian Motion on a Circle", fontsize=28)

    ax[2].plot(sim_mil[0], sim_mil[1], label="Milstein", color='r')
    # ax[2].plot(ana_circle[0], ana_circle[1], label="Unit Circle", color="black", linestyle="--")
    ax[2].plot(t, ana_hyper[0], label="Hyperbola", color="black", linestyle="--")
    ax[2].plot(t, ana_hyper[1], color="black", linestyle="--")
    ax[2].set_xlabel("x", fontsize=22)
    ax[2].set_ylabel("y", fontsize=22)
    ax[2].legend(fontsize=14)
    ax[2].set_aspect('equal', 'box')

    """
    errors = list()
    level = func.surface(start[0], start[1])
    for i in range(len(sim_euler[0])):
        e = func.surface(sim_euler[0][i], sim_euler[1][i]) - level
        errors.append(e)

    print("Errors Counted")
    counts, bins = np.histogram(errors, 50)

    fig, ax = plt.subplots()

    ax1 = plt.subplot(121)
    ax1.plot(sim_euler[0], sim_euler[1], label="Euler Method")
    ax1.plot(ana_circle[0], ana_circle[1], linestyle="--", color="black", label="Unit Circle")
    ax1.set_title("Brownian Motion on a Circle", fontsize=28)
    ax1.set_xlabel("x", fontsize=22)
    ax1.set_ylabel("y", fontsize=22)
    ax1.legend(fontsize=14)
    ax1.axis('equal')

    ax2 = plt.subplot(224)
    ax2.hist(bins[:-1], bins, weights=counts, rwidth=0.8)
    ax2.set_title(r"Histogram of the Deviance", fontsize=28)
    ax2.set_xlabel(r"Size", fontsize=22)
    ax2.set_ylabel(r"Frequency", fontsize=22)

    ax3 = plt.subplot(222)
    ax3.vlines(deviates, np.min([np.min(sim_euler[0]), np.min(sim_euler[1])]),
               np.max([np.max(sim_euler[0]), np.max(sim_euler[1])]), color="r")
    ax3.plot(h_vec, sim_euler[0], label="x coordinate")
    ax3.plot(h_vec, sim_euler[1], label='y coordinate', color="orange")
    ax3.set_title(r"Components of the Brownian Motion, \(\varepsilon=0.01\)", fontsize=28)
    ax3.set_xlabel(r"Time, \(t\)", fontsize=22)
    ax3.set_ylabel(r"Position", fontsize=22)
    ax3.legend(fontsize=14)

    plt.subplots_adjust(left=0.06, bottom=0.064, right=0.952, top=0.945, wspace=0.2, hspace=0.279)

    plt.show()


def surface_plotter():
    n = 100
    h = np.linspace(0.01, 0.1, n)
    e = np.linspace(0.01, 0.1, n)
    critical_e = h - 0.25 * np.power(h, 2)
    z = np.zeros(n)

    probs = np.zeros((n, n))

    for i in range(len(h)):
        for j in range(len(e)):
            zero = np.sqrt(e[j] + h[i] - 0.25 * h[i] ** 2)
            test = np.sqrt(-e[j] + h[i] - 0.25 * h[i] ** 2)

            probs[i, j] += 1 - sp.norm.cdf(zero, loc=0, scale=np.sqrt(h[i])) + \
                           sp.norm.cdf(-zero, loc=0, scale=np.sqrt(h[i]))

            if not math.isnan(test):
                probs[i, j] += sp.norm.cdf(test, loc=0, scale=np.sqrt(h[i])) - \
                               sp.norm.cdf(-test, loc=0, scale=np.sqrt(h[i]))

    h_mesh, e_mesh = np.meshgrid(h, e)

    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(h_mesh, e_mesh, probs)

    ax.plot3D(h, critical_e, z, color='red')
    ax.set_xlabel(r"Step-size, \(h\)")
    ax.set_ylabel(r"Tolerance, \(\varepsilon\)")
    ax.set_zlabel(r"Error Probability")

    plt.title(r"One-Step Error Probability Surface")
    plt.show()


def triple_circle():
    np.random.seed(270300)

    m = 1000
    theta = np.linspace(0, 2*np.pi, m)

    ana_circle = np.zeros((2, m))

    for i in range(m):
        ana_circle[0][i] = np.cos(theta[i])
        ana_circle[1][i] = np.sin(theta[i])

    h = 0.0001
    n = int(10/h)
    white_noise = schemes.generate_wiener_diff(h, n)

    h_vec = np.zeros(n+1)
    for i in range(n):
        h_vec[i+1] = h_vec[i] + h

    func = Circle()
    start = [1, 0]
    scheme = schemes.euler

    sim_euler = np.zeros((2, n + 1))
    sim_euler[0][0] = start[0]
    sim_euler[1][0] = start[1]

    sim_rk = np.zeros((2, n + 1))
    sim_rk[0][0] = start[0]
    sim_rk[1][1] = start[1]

    sim_mil = np.zeros((2, n + 1))
    sim_mil[0][0] = start[0]
    sim_mil[1][0] = start[1]

    for i in range(len(white_noise)):
        x = sim_euler[0][i]
        y = sim_euler[1][i]

        mu_x = func.mu_x(x, y)
        mu_y = func.mu_y(x, y)
        sigma_x = func.diff_x(x, y)
        sigma_y = func.diff_y(x, y)

        x_new = scheme(x, mu_x, sigma_x, h, white_noise[i])
        y_new = scheme(y, mu_y, sigma_y, h, white_noise[i])

        sim_euler[0][i + 1] = x_new
        sim_euler[1][i + 1] = y_new

        x = sim_rk[0][i]
        y = sim_rk[1][i]

        mu_x = func.mu_x(x, y)
        mu_y = func.mu_y(x, y)
        sigma_x = func.diff_x(x, y)
        sigma_y = func.diff_y(x, y)
        supp = func.support(x, y, h)

        x_new = schemes.runge_kutta(x, mu_x, sigma_x, supp[0], h, white_noise[i])
        y_new = schemes.runge_kutta(y, mu_y, sigma_y, supp[1], h, white_noise[i])

        sim_rk[0][i + 1] = x_new
        sim_rk[1][i + 1] = y_new

        x = sim_mil[0][i]
        y = sim_mil[1][i]

        mu_x = func.mu_x(x, y)
        mu_y = func.mu_y(x, y)
        sigma_x = func.diff_x(x, y)
        sigma_y = func.diff_y(x, y)
        derivative = func.derivative(x, y)

        x_new = schemes.milstein(x, mu_x, sigma_x, derivative[0], h, white_noise[i])
        y_new = schemes.milstein(y, mu_y, sigma_y, derivative[1], h, white_noise[i])

        sim_mil[0][i + 1] = x_new
        sim_mil[1][i + 1] = y_new

    abs_euler = 0
    abs_rk = 0
    abs_mil = 0

    for i in range(len(white_noise)):
        abs_euler += abs(func.surface(sim_euler[0][i], sim_euler[1][i]) - 1)
        abs_mil += abs(func.surface(sim_mil[0][i], sim_mil[1][i]) - 1)
        abs_rk += abs(func.surface(sim_rk[0][i], sim_rk[1][i]) - 1)

    print("Euler:", abs_euler)
    print("Rk:", abs_rk)
    print("Mil", abs_mil)

    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    fig, ax = plt.subplots(1, 3)

    ax[0].plot(sim_euler[0], sim_euler[1], label="Euler Method")
    ax[0].plot(ana_circle[0], ana_circle[1], label="Unit Circle", color="black", linestyle="--")
    ax[0].set_xlabel("x", fontsize=22)
    ax[0].set_ylabel("y", fontsize=22)
    ax[0].legend(fontsize=14)
    ax[0].set_aspect('equal', 'box')

    ax[1].plot(sim_rk[0], sim_rk[1], label="Runge-Kutta", color='orange')
    ax[1].plot(ana_circle[0], ana_circle[1], label="Unit Circle", color="black", linestyle="--")
    ax[1].set_xlabel("x", fontsize=22)
    ax[1].set_ylabel("y", fontsize=22)
    ax[1].legend(fontsize=14)
    ax[1].set_aspect('equal', 'box')
    ax[1].set_title("Three Simulations of the Brownian Motion on a Circle", fontsize=28)

    ax[2].plot(sim_mil[0], sim_mil[1], label="Milstein", color='r')
    ax[2].plot(ana_circle[0], ana_circle[1], label="Unit Circle", color="black", linestyle="--")
    ax[2].set_xlabel("x", fontsize=22)
    ax[2].set_ylabel("y", fontsize=22)
    ax[2].legend(fontsize=14)
    ax[2].set_aspect('equal', 'box')

    plt.show()


intro_errors()
# input_vs_actual()
# triple_circle()
# surface_plotter()
