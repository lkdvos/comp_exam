
from functions import gaussian_integral
from constants import *
import numpy as np
from calc_spectrum import calc_spectrum
import mpl_toolkits.mplot3d.axes3d as p3
import pylab as pl

def exp_val_matrices(nf, a=a):

    # matrices initialiseren, respectievelijk de matrixelementen van x, x², px en px²
    x_matrix = np.zeros((nf,nf),dtype=np.float64)
    x2_matrix = np.zeros((nf,nf),dtype=np.float64)
    px_matrix = np.zeros((nf,nf),dtype=np.float64)
    px2_matrix = np.zeros((nf,nf),dtype=np.float64)

    for m in np.arange(nf):
        for n in np.arange(nf):
            x_matrix[n,m] = gaussian_integral(a, n+m+1)
            x2_matrix[n,m] = gaussian_integral(a, n+m+2)
            px_matrix[n,m] = n * gaussian_integral(a, n+m-1) - a * gaussian_integral(a, n+m+1)
            px2_matrix[n,m] = -a**2 * gaussian_integral(a, m + n + 2) + (2*a*n + a) * gaussian_integral(a, m+n) -(n**2 - n) * gaussian_integral(a, m+n-2)


    return x_matrix, x2_matrix, px_matrix, px2_matrix


def calc_heisenberg(nf, V0, alpha, n):
    # matrixelementen van de relevante operatoren in de gebruikte basis berekenen
    a = np.power(V0, 0.5) * alpha
    E_vals, E_vecs = calc_spectrum(nf, a, alpha, V0)

    x_matrix, x2_matrix, px_matrix, px2_matrix = exp_val_matrices(nf, a)

    exp_val_x = np.einsum('k,l,kl', E_vecs[:,n], E_vecs[:,n], x_matrix)
    exp_val_x2 = np.einsum('k,l,kl', E_vecs[:,n], E_vecs[:,n], x2_matrix)
    exp_val_px = np.einsum('k,l,kl', E_vecs[:,n], E_vecs[:,n], px_matrix)
    exp_val_px2 = np.einsum('k,l,kl', E_vecs[:,n], E_vecs[:,n], px2_matrix)

    delta_x = np.sqrt(exp_val_x2 - exp_val_x ** 2)
    delta_px = np.sqrt(exp_val_px2 - exp_val_px ** 2)

    return delta_x * delta_px

def plot_heisenberg():

    X, Y = np.meshgrid(V0_range, alpha_range)

    matrix_3Dplot = np.zeros((V0_steps, alpha_steps, n))
    for i, V0 in enumerate(V0_range):
        for j, alpha in enumerate(alpha_range):
            for _n in range(n):
                matrix_3Dplot[i][j][_n] = calc_heisenberg(nf, V0, alpha, _n)

    fig = pl.figure()
    ax = fig.gca(projection='3d')
    print(X.shape, Y.shape, matrix_3Dplot.shape, matrix_3Dplot.shape)


    for _n in range(n-1):
        ax.plot_wireframe(X, Y, matrix_3Dplot[:,:,_n], label = "{}".format(_n), color="{}".format(_n/n))
    ax.legend()
    ax.set_xlabel('V0')
    ax.set_ylabel('alpha')
    pl.show()

plot_heisenberg()
