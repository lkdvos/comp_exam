import math as m
from scipy.integrate import quad, simps
import numpy as np
import pylab as pl


# Define the program constants

Re = 4 # centrum potentiaal
alpha = 0.3
R = 20 # centrum golfpakket op t = 0
b = 2 # breedte golfpakket op t = 0

# Define constants for reading wavefunctions
Start = Re / 10000
HStep = 0.005
MaxDist = 5 / alpha
MaxI = int(m.ceil((MaxDist-Start)/HStep))

MinEner, MaxEner = 0.08, 16
k_steps = 100
k_list = np.linspace(MinEner**0.5, MaxEner**0.5, k_steps) # make list of linearly spaced k_values

fwavefunc = np.loadtxt('exam_Comp/part2/wavefunc-k.txt') # import wavefunctions

 # rewrite wavefunctions in more handlable format

# WAVEFUNCTIONS[E_i][j] gives you the u_j eigenfunction with incoming energy of E_i
WAVEFUNCTIONS = dict()
rlen = MaxI//3+1 # amount of r_points
r = fwavefunc[0:rlen, 1] # Alle r-waarden (heeft zelfde shape als WAVEFUNCTIONS[E_i][j])

for k in range(k_steps):
  WAVEFUNCTIONS[k] = [[] for i in range(6)]
  for u in range(6): # implement first 6 L-eigenfunctions
    WAVEFUNCTIONS[k][u] = fwavefunc[k*rlen:(k+1)*rlen, 2+u]

# create Phi as in section IV of the article:
# Phi[k, r] is L=0 eigenfunction for incident energy corresponding to k, at value r
Phi = np.zeros((100,rlen))
for i in range(k_steps):
    for j in range(rlen):
        Phi[i,j] = WAVEFUNCTIONS[i][0][j]


# create Psi(r, 0) as in section IV eq. (31) of the article
# works also for entire list of r
def Psi(r, R, b):
    return np.exp(-1* np.power(r-R, 2) * np.power(b, 2))

Psi = Psi(r, R, b)


# calculate C(k) as in section IV eq. (29)
def findC(r, Phi, Psi):
    integrandum = Phi * Psi # element-wise multiplication, each element corresponding to r value
    return 2 / np.pi * simps(integrandum, r)


# calculate the general wave packet at different times as eq. (27)
def full_wave(r_, t):
    integrandum1 = np.exp(-j*np.power(k_list, 2) * t)
    integrandum2 = Phi[:,r_]

    # make list of C_k for different k_values
    C_k = [findC(r, Phi[k,:], Psi) for k in range(k_steps)]

    # integrate over k_values
    integrandum = C_k * integrandum1 * integrandum2
    return simps(integrandum, k_list)

# calculate the resonant part of the wave packet as eq. (32)
# should be done around resonance-peak k_r with width delta
# WIP, this has not been fully implemented
def resonant_wave(r, t, k_r, delta):
    # limit k-values over which to integrate, see what indexes:
    k_list = np.linspace(k_r - delta, k_r + delta, dk)
    k_list_index = []

    # create integrands
    j = complex(0,-1)
    integrandum1 = np.exp(j*np.power(k_list, 2) * t)
    integrandum2 = Phi[k_list_index,r]

    # make list of C_k for different k_values
    C_k = [findC(r, Phi[k,:], Psi) for k in k_list_index]

    integrandum = C_k * integrandum1 * integrandum2
    return simps(integrandum, k_list)


if __name__ == "__main__":
    # create and plot C_k
    C_k = [findC(r, Phi[k,], Psi) for k in range(100)]
    pl.figure()
    pl.plot(k_list, C_k, label='C_k')
    pl.legend()
    pl.xlabel('k (fm^-2)')
    pl.ylabel('C_k')
    pl.title('C_k for a wavepacket')


    # create and plot Psi(r, t) for different values of t
    t_max = 2
    t_steps = 5
    t = np.linspace(0, t_max, 5)

    Psi_tot = [[full_wave(r_, t_) for r_ in range(rlen)] for t_ in t]

    pl.figure()

    for i, t_ in enumerate(t):
        pl.plot(r, Psi_tot[i], label='t={}'.format(t_))

    pl.legend()
    pl.xlabel('r')
    pl.ylabel('Psi(r, t)')
    pl.title('General wavepackets at different times')
    pl.legend()
    pl.show()

'''
Even though the program runs, we are unsure if the implementation is correct.
We expect that the general wavepacket at t=0 should give the gaussian function, however we do not seem to find this behaviour.
This could be due to faulty implementation, or because of the small range of data we have for the different k values,
or because our wavefunctions aren't correctly formed. Further testing is necessary with larger datasets in order to figure this out.
'''
