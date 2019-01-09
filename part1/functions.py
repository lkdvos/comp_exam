
from constants import *
from scipy.special import gamma, comb
import numpy as np


def gaussian_integral(a, n):
    # returns value of definite integral -inf +inf for exp(-ax²)x^n
    if n % 2 != 0:
        return 0.
    else:
        return gamma( (n+1.)/2.) / (a**((n+1.)/2.) )

def gaussian_integral_ext(a, b, n):
    # returns value of definite integral -inf + inf for exp(-ax^2 - 2bx)x^n
    out = 0
    for k in range(n+1):
        out += np.power(b/a, n-k) * comb(n, k) * gaussian_integral(a, k)

    out *= np.exp(np.power(b, 2) / a)
    return out

def morse(x, alpha, V0):
    # de morsepotentiaal in functie van x, alpha, V0
    # V0 is positief gedefinieerd voor negatieve
    return - V0 * np.exp(-x * alpha) * (2 - np.exp(-x * alpha))

def bf_qm_ho(x, n, a):
    # basisfuncties van de kwantum harmonische oscillator (niet orthonormaal)
    return x ** n * np.exp(- a * x ** 2 / 2)
