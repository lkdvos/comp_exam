#####################################################
# script used to answer the question BOUND2 in part 1 of the project
#
# we calculate the Heisenberg uncertainty product by determining the relevant expectation values
# of the position and momentum operator for the calculated eigenfunctions
#
# we do this by first determining these matrixelement in terms of the basisfunction
# and the use the coefficients of the eigenfunctions in this basis to calculate the expectation values
#
# the basis functions are defined in the script functions.py
#
#####################################################


from functions import gaussian_integral
from constants import *
import numpy as np
from calc_spectrum import calc_spectrum
import mpl_toolkits.mplot3d.axes3d as p3
import pylab as pl

def exp_val_matrices(nf, a=a):
    ''' Determine and return the matrix elements of the operators x, x**2, px and px**2
    in terms of the basis functions '''

    # initialize the matrices
    x_matrix = np.zeros((nf,nf),dtype=np.float64)
    x2_matrix = np.zeros((nf,nf),dtype=np.float64)
    px_matrix = np.zeros((nf,nf),dtype=np.float64)
    px2_matrix = np.zeros((nf,nf),dtype=np.float64)

    for m in np.arange(nf):
        for n in np.arange(nf):
            # use the calculated analytical expressions to determine the matrix elements
            x_matrix[n,m] = gaussian_integral(a, n+m+1)
            x2_matrix[n,m] = gaussian_integral(a, n+m+2)
            px_matrix[n,m] = n * gaussian_integral(a, n+m-1) - a * gaussian_integral(a, n+m+1)
            px2_matrix[n,m] = -a**2 * gaussian_integral(a, m + n + 2) + (2*a*n + a) * gaussian_integral(a, m+n) -(n**2 - n) * gaussian_integral(a, m+n-2)

    # return the matrices
    return x_matrix, x2_matrix, px_matrix, px2_matrix


def calc_heisenberg(nf, V0, alpha, n):
    ''' Return the product Deltax * Deltapx for the first n eigenfunctions,
    determined variationally for a basis of size nf '''

    a = np.power(V0, 0.5) * alpha

    # determine the energy eigenvalues and eigenvectors for the given combination
    # of th potential parameters V0 and alpha
    E_vals, E_vecs = calc_spectrum(nf, alpha, V0, a)

    x_matrix, x2_matrix, px_matrix, px2_matrix = exp_val_matrices(nf, a)

    # initialize an array for the uncertainty products for the first n wavefunctions
    uncertainty_product = np.zeros(n)

    for _n in range(n):
        exp_val_x = np.einsum('k,l,kl', E_vecs[:,_n], E_vecs[:,_n], x_matrix)
        exp_val_x2 = np.einsum('k,l,kl', E_vecs[:,_n], E_vecs[:,_n], x2_matrix)
        exp_val_px = np.einsum('k,l,kl', E_vecs[:,_n], E_vecs[:,_n], px_matrix)
        exp_val_px2 = np.einsum('k,l,kl', E_vecs[:,_n], E_vecs[:,_n], px2_matrix)

        delta_x = np.sqrt(exp_val_x2 - exp_val_x ** 2)
        delta_px = np.sqrt(exp_val_px2 - exp_val_px ** 2)

        uncertainty_product[_n] = delta_x * delta_px

    return uncertainty_product

def plot_heisenberg():
    ''' Plots the product Deltax * Deltapx for the first 3 calculated wavefunctions
    for a range of different potential parameters '''

    # a grid of the different combinations of the potential parameters
    X, Y = np.meshgrid(V0_range, alpha_range)

    # initialize and then fill the matrix with the different Deltax * Deltapx values
    # matrix_3Dplot[i,j,n] contains the value of Deltax * Deltapx for the n-th
    # calculated wavefunction for the combination of
    # potential parameters V0 = V0_range[i] and alpha = alpha-range[j]
    matrix_3Dplot = np.zeros((V0_steps, alpha_steps, n))
    for i, V0 in enumerate(V0_range):
        for j, alpha in enumerate(alpha_range):
            heis = calc_heisenberg(nf, V0, alpha, n)
            for _n in range(n):
                matrix_3Dplot[i][j][_n] = heis[_n]

    fig = pl.figure()
    ax = fig.gca(projection='3d')


    for _n in range(n):
        ax.plot_wireframe(X, Y, matrix_3Dplot[:,:,_n], label = "{}".format(_n), color="{}".format(_n/n))
    ax.legend()
    ax.set_xlabel('V0')
    ax.set_ylabel('alpha')
    ax.set_zlabel(r'$\Delta x \Delta p_x$')
    pl.show()

if __name__ == "__main__":

    plot_heisenberg()
