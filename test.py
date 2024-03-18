import mystic.constraints
from mystic.solvers import diffev2, buckshot
from mystic.monitors import VerboseMonitor


def stabilise(x_0, surf, level):
    def dist_min(x):
        return (x[0] - x_0[0])**2 + (x[1] - x_0[1])**2

    def constraint(x):
        return surf(x[0], x[1]) - level

    @mystic.penalty.lagrange_equality(constraint, k=1000)
    def penalty(x):
        return 0

    bounds = [(0, 10000)]*2
    mon = VerboseMonitor(10)

    result = diffev2(dist_min, x0=x_0, penalty=penalty, bounds=bounds, npop=100, gtol=200, disp=False, full_output=True,
                     itermon=mon, ftol=0.0000005)

    return result


def surface(x, y):
    return x*y


res = stabilise([1.01, 1.01], surface, 1)
print(surface(res[0][0], res[0][1]))
