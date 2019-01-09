from constants import *
import pylab as pl
from functions import morse, bf_qm_ho
from calc_spectrum import calc_spectrum, write_wf


def plot_wavef(nf=nf, alpha=alpha, V0=V0):
    xrange = 3/alpha
    yrange = V0
    x = np.linspace(-xrange, xrange, 100)
    E_vals, E_vecs = calc_spectrum(nf, alpha, V0)
    write_wf(n, E_vecs, x, bf_qm_ho, "wavef.txt")
    data = np.loadtxt("wavef.txt")

    pl.figure()
    pl.xlabel('distance (alpha*x)')
    pl.ylabel('Energy')
    pl.title('Calculated wavefunctions')
    # plot energy values
    for _n in range(n):
        pl.plot(data[:,0], E_vals[_n] * np.ones(100))

    # plot wavefunctions translated to their corresponding energy values
    for _n in range(n):
        pl.plot(data[:,0], wavefactor * data[:,1+_n] + E_vals[_n], label="wave{}".format(_n))

    # plot potential
    pl.plot(data[:,0], morse(data[:,0], alpha, V0), linestyle="--")

    pl.legend()
    pl.xlim(-xrange, xrange)
    pl.ylim(-yrange, 0)
    pl.show()


def plot_E_convergence(nf=nf, n=n, alpha=alpha, V0=V0):

    E_vals_list = [[] for _ in range(nf)]

    for _nf in range(1, nf):
        E_vals, E_vecs = calc_spectrum(_nf, alpha, V0)
        for _n in range(_nf):
            E_vals_list[_n].append(E_vals[_n])

    pl.figure()
    pl.xlabel('number of basis functions')
    pl.ylabel('Energy(unit)')
    pl.title('Energy convergence for increasing basis set')

    for _n in range(1, n):
        print(_n)
        pl.plot(range(_n, nf), E_vals_list[_n-1], 'o-', label='Energy level {}'.format(_n-1))

    pl.legend()
    pl.show()

if __name__ == "__main__":
    plot_wavef(nf, 0.5, 100)
    plot_wavef(nf, 0.5, 1000)
    plot_E_convergence(nf, n)
