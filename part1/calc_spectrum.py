from functions import gaussian_integral, gaussian_integral_ext
from constants import *
from scipy.linalg import eigh
import numpy as np
import pylab as pl

def calc_spectrum(nf, alpha=alpha, V0=V0):
    # matrices initialiseren, respectievelijk de overlap, kinetische- en potentiële energie
    a = np.power(V0, 0.5)*alpha
    S_matrix = np.zeros((nf,nf),dtype=np.float64)
    T_matrix = np.zeros((nf,nf),dtype=np.float64)
    V_matrix = np.zeros((nf,nf),dtype=np.float64)
    for m in np.arange(nf):
        for n in np.arange(nf):
            S_matrix[m,n] = gaussian_integral(a, m + n)
            T_matrix[m,n] = -a**2 * gaussian_integral(a, m + n + 2) + (2*a*n + a) * gaussian_integral(a, m+n) -(n**2 - n) * gaussian_integral(a, m+n-2)
            V_matrix[m,n] = 2 *-V0 * gaussian_integral_ext(a, alpha / 2, m + n) + V0 * gaussian_integral_ext(a, alpha, m + n)
    H_matrix = T_matrix + V_matrix

    return eigh(H_matrix, S_matrix)

def write_wf(n, v, x_values, basisf, filename):
    # maximaal aantal golffuncties dat uitgeschreven mag worden
    num_max = 10
    if(n < num_max):
        num_max = n
    f = open(filename,"w")

    for x in x_values:
        f.write("{}".format(x))  # write the x value
        for i in range(num_max): # for this x, write the first num_max wavefunctions evaluated in x
            t=0.
            for j in range(len(v)):  # loop over rows containing coefficients
                t += v[j,i] * basisf(x, j, a)
            f.write("\t{}".format(t))
        f.write("\n") # go to a new line
    f.close() # close the file after the loop over all x values
