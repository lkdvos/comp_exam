#####################################################
# script used to plot the results obtained in calc_spectrum.py
#####################################################
from constants import *
import pylab as pl
from functions import morse, bf_qm_ho
from analy import l, E_val_analy
from calc_spectrum import calc_spectrum, write_wf


def plot_wavef(nf=nf, alpha=alpha, V0=V0):
    ''' Routine to plot the wavefunctions obtained through variational calulus
    with a basis of nf functions '''

    # determine the ranges for the axes
    xrange = 3/alpha
    yrange = V0

    x = np.linspace(-xrange, xrange, 100) * alpha

    # calculate the eigenvalues and eigenvactors using the variational method
    E_vals, E_vecs = calc_spectrum(nf, alpha, V0)

    # write the first n (as defined in constants.py) wavefunctions to the file wavef.txt
    write_wf(n, E_vecs, x, bf_qm_ho, "wavef.txt")

    # load the data for the plot
    data = np.loadtxt("wavef.txt")

    pl.figure()
    pl.xlabel('distance (alpha*x)')
    pl.ylabel('Energy')
    pl.title('Calculated wavefunctions')
    # plot energy values
    for _n in range(n):
        pl.plot(data[:,0], E_vals[_n] * np.ones(100))

    # plot wavefunctions translated to their corresponding energy values
    # the factor wavefactor is purely for visualization
    for _n in range(n):
        pl.plot(data[:,0], wavefactor * data[:,1+_n] + E_vals[_n], label="wavefunction {}".format(_n))

    # plot Morse potential for reference
    pl.plot(data[:,0], morse(data[:,0], alpha, V0), linestyle="--")

    pl.legend()
    pl.xlim(-xrange, xrange)
    pl.ylim(- yrange, 0)
    pl.show()


def plot_E_convergence(nf=nf, n=n, alpha=alpha, V0=V0):
    ''' Routine to plot the calculated energy eigenvalues as a function of
    the number of basis functions in the basis set '''

    E_vals_list = [[] for _ in range(nf)]

    for _nf in range(1, nf):
        E_vals, E_vecs = calc_spectrum(_nf, alpha, V0)
        for _n in range(_nf):
            E_vals_list[_n].append(E_vals[_n])

    pl.figure()
    pl.xlabel('number of basis functions')
    pl.ylabel('Energy')
    pl.title('Energy convergence for increasing basis set')

    for _n in range(1, n + 1):
        pl.plot(range(_n, nf), E_vals_list[_n-1], 'o-', label='Energy level {}'.format(_n-1))
        pl.plot(range(_n, nf), [E_val_analy(_n - 1) for k in range(nf-_n)], linestyle = '--')

    pl.legend()
    pl.show()

if __name__ == "__main__":
    plot_wavef(nf, 1., 100)
    plot_E_convergence(nf, n)
