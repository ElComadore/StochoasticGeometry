import numpy as np
import generator
import matplotlib.pyplot as plt


def polar_coord(r_0, theta_0, h, w_1, w_2):
    r = list()
    r.append(r_0)
    theta = list()
    theta.append(theta_0)

    for i in range(len(w_1)):
        r_new = r[-1] + (1/(2*r[-1])) * h + np.cos(theta[-1]) * w_1[i] + np.sin(theta[-1]) * w_2[i]
        theta_new = theta[-1] - (1/r[-1]) * np.sin(theta[-1]) * w_1[i] + (1/r[-1]) * np.cos(theta[-1]) * w_2[i]

        r.append(r_new)
        theta.append(theta_new)

    return r, theta


def main():
    n = 1000000
    h = 1/n
    start = 0.0001

    w_1 = generator.generate_wiener_diff(h, n)
    w_2 = generator.generate_wiener_diff(h, n)

    a = list()
    b = list()
    a.append(start)
    b.append(0)

    for i in range(len(w_1)):
        a_new = a[-1] + w_1[i]
        b_new = b[-1] + w_2[i]

        a.append(a_new)
        b.append(b_new)

    r_0 = start
    theta_0 = 0

    r, theta = polar_coord(r_0, theta_0, h, w_1, w_2)

    x = np.multiply(r, np.cos(theta))
    y = np.multiply(r, np.sin(theta))

    plt.plot(x, y)
    plt.plot(a, b, linestyle='--')
    plt.show()


main()
