# script with all the constants used in this part of the project

# we always work in units where hbar ** 2 / (2 * m) = 1

import numpy as np

# n: the number of energy eigenfunctions to be determined through variational calculus
n = 4

# nf: the number of basis functions in the basis set
nf = 5

# the depth of the Morse potential
V0 = 100.

# a range for plots of the potential, dependent on the depth
yrange = V0 * 1

# factor alpha in the definition of the Morse potential
# 1 / alpha determines the width of the potential
alpha = 1

# a range for plots of the potential
xrange = 3. / alpha

# parameter that determines the width of the basis functions in the basis set
# taken so that (in units hbar ** 2 / (2 * m) = 1) the factor a
# corresponds with the factor in the gaussian exponent of the quantom harmonic oscillator
# that best approximates the potential for a given set of parameters
a = np.sqrt(V0) * alpha

# a multiplication factor for plots of the eigenfunctions
# used to visualize the eigenfunctions on a plot with the potential
wavefactor = 8

# number of steps for loops over different values of the potential parameters for use in heisenberg.py
V0_steps, alpha_steps = 30, 30

# limits for loops over different values of the potential parameters for use in heisenberg.py
V0_min, V0_max = .1, 1000
alpha_min, alpha_max = .1, 10

# the ranges for loops over different values of the potential parameters for use in heisenberg.py
V0_range = np.linspace(V0_min, V0_max, V0_steps)
alpha_range = np.linspace(alpha_min, alpha_max, alpha_steps)
