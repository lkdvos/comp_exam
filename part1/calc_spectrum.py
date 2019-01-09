#####################################################
# script used to answer the question BOUND1 in part 1 of the project
#
# in plotters.py the results of this script are plotted
#
# we use a basis set similar to the eigenfunctions of the one-dimensional
# quantum harmonic oscillator, where we have replaced the Hermite polynomial Hn
# by the factor x ** n to simplify the analytical results of the integrals
#
# the basis functions are defined in the script functions.py
#
# the form of the gaussian exponent (factor a) was determined by expanding the
# Morse potential in a Taylor series around the equilibrium point x = 0
# and determining the harmonic constant k in terms of the potential parameters
# this value was then plugged into the formula for a found on https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
# all quantities were converted units where hbar ** 2 / (2 * m) = 1
#
#
# This script is based on the program Deepwell.py from the theory lectures
#
#####################################################

from functions import gaussian_integral, gaussian_integral_ext
from constants import *
from scipy.linalg import eigh
import numpy as np
import pylab as pl

def calc_spectrum(nf, alpha = alpha, V0 = V0, a = a):
    ''' A routine to build the overlap and Hamiltonian matrix for the first nf basis functions,
    and solve the variational eigenvalue problem

    returns the nf eigenvalues and the coefficients of the nf eigenfunctions
    in terms of the basis set '''

    # initialize the overlap, kinetic- and potential energy matrices
    a = np.power(V0, 0.5)*alpha

    # initialize the overlap, kinetic- and potential energy matrices
    S_matrix = np.zeros((nf,nf),dtype=np.float64)
    T_matrix = np.zeros((nf,nf),dtype=np.float64)
    V_matrix = np.zeros((nf,nf),dtype=np.float64)

    # fill the matrices
    for m in np.arange(nf):
        for n in np.arange(nf):

            # use the found analytical expressions for the matrixelements
            S_matrix[m,n] = gaussian_integral(a, m + n)
            T_matrix[m,n] = -a**2 * gaussian_integral(a, m + n + 2) + (2*a*n + a) * gaussian_integral(a, m+n) -(n**2 - n) * gaussian_integral(a, m+n-2)
            V_matrix[m,n] = 2 *-V0 * gaussian_integral_ext(a, alpha / 2, m + n) + V0 * gaussian_integral_ext(a, alpha, m + n)

    H_matrix = T_matrix + V_matrix

    # solve the generalised eigenvalue problem and return the eigenvalues and eigenvectors
    return eigh(H_matrix, S_matrix)

def write_wf(n, v, x_values, basisf, filename):
    ''' Routine to write the first n wavefunctions to a file given the coefficient
    matrix v, an array of x values, the definition of the basis functions and the namen of the file '''

    # maximum number of wavefunctions that can be written to the file
    num_max = nf
    if(n < num_max):
        num_max = n
    # open the output file
    f = open(filename,"w")

    for x in x_values:
        # write the x value
        f.write("{}".format(x))
        for i in range(num_max): # for this x, write the first num_max wavefunctions evaluated in x
            t=0.
            # for this x, write the first num_max wavefunctions evaluated in x
            for j in range(len(v)):
                # v[j,i] is the coefficient of basisfunction j in the basis decomposition of eigenvector i
                t += v[j,i] * basisf(x, j, a)
            f.write("\t{}".format(t))
        f.write("\n")
    f.close()
