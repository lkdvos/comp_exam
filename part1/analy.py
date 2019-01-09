# this module is used to analytically calculate the energy eigenvalues
# and eigenfunctions of the Morse potential
#
# all analytical expressions were taken from https://en.wikipedia.org/wiki/Morse_potential
# all quantities were converted units where hbar ** 2 / (2 * m) = 1

import numpy as np
from scipy.special import assoc_laguerre, gamma
import pylab as pl
from constants import *
from functions import morse

# parameters of the Morse potential where imported from constants.py
# define the dimensionless constant l in terms of these parameters
l = np.sqrt(V0) / alpha

def E_val_analy(n):
    ''' Return the n-th eigenvalue of the Morse potential
    with parameters V0 and alpha as given in constants.py'''

    e = np.power(l, 2) - np.power((l - n - 0.5), 2)
    return e * np.power(alpha, 2) - V0

def wf_analy(x, n):
    ''' Return the value of the n-th eigenfunction of the
    1-dimensional SE for the Morse potential, evaluated in x'''

    # normalization factor
    Norm = gamma(n + 1) * (2 * l - 2 * n - 1)
    Norm /= gamma(2 * l - n)
    Norm = np.power(Norm, 0.5)

    # help variable
    z = 2 * l * np.exp(-alpha * x)

    return Norm * np.power(z, l - n -0.5) * np.exp(-0.5 * z) * assoc_laguerre(z, n, 2 * l - 2 * n - 1)

def plot_analywavef(n, alpha=alpha, V0=V0):
    ''' A routine to plot the first n eigenfunctions and eigenvalues
    for the Morse potential '''

    # initialize the figure
    pl.figure()

    # the xrange is dependent on alpha, the scale of the potential
    # imported from constants.py
    x = np.linspace(-xrange,xrange,100)

    # plot the first n eigenfunctions (starting the count from zero)
    #
    # the y-axis is the energy axis, and the morse potential with the corresponding
    # parameters is plotted for reference
    #
    # the eigenfunctions are centered on their corresponding energy eigenvalue
    # and are multiplied by a factor 4 purely for visualisation
    #
    for _n in range(n):
        pl.plot(x, 5 * wf_analy(x,_n) + E_val_analy(_n), label="{}".format(_n))

    pl.plot(x, morse(x, alpha, V0), linestyle='--')


    # yrange depends on the depth V0 of the potential
    pl.ylim(-1*yrange,0)
    pl.xlim(-xrange, xrange)
    pl.xlabel(r'$x$')
    pl.ylabel(r'Energy')
    pl.legend()
    pl.title('The first {} analytical wavefunctions for the Morse potential'.format(n))
    pl.show()

if __name__ == "__main__":

    # plot the first 7 eigenfunctions and eigenvalues for the Morse potential
    # with parameters as defined in constants.py
    plot_analywavef(7)
