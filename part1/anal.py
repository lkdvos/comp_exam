# ANALYTISCHE OPLOSSINGEN
import numpy as np
from scipy.special import assoc_laguerre, gamma
import pylab as pl
from constants import *
from functions import morse

l = np.sqrt(V0) / alpha


def E_val_anal(n):
    e = np.power(l, 2) - np.power((l - n - 0.5), 2)
    return e * np.power(alpha, 2) - V0

def wf_analy(x, n):
    Norm = gamma(n + 1) * (2 * l - 2 * n - 1)
    Norm /= gamma(2 * l - n)
    Norm = np.power(Norm, 0.5)

    z = 2 * l * np.exp(-alpha * x)

    return wavefactor * Norm * np.power(z, l - n -0.5) * np.exp(-0.5 * z) * assoc_laguerre(z, n, 2 * l - 2 * n - 1) + E_val_anal(n)

def plot_analwavef(n, alpha=alpha, V0=V0):
    pl.figure()
    x = np.linspace(-xrange,xrange,100)



    for _n in range(n):
        pl.plot(x, wf_analy(x,_n), label="wave{}".format(_n))
    pl.plot(x, morse(x, alpha, V0), linestyle='--')

    pl.ylim(-1*yrange,0)
    pl.xlim(-xrange, xrange)
    pl.legend()
    pl.title('first {} analytical wavefunctions of Morse'.format(n))
    pl.show()

if __name__ == "__main__":
    plot_analwavef(10)
