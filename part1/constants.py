import numpy as np

n = 5
nf = 12



V0 = 100.
yrange = V0 * 1
alpha = .5
xrange = 3. / alpha
a = np.sqrt(V0) * alpha
wavefactor = 1


V0_steps, alpha_steps = 30, 30

V0_min, V0_max = .1, 1000
alpha_min, alpha_max = .1, 10

V0_range = np.linspace(V0_min, V0_max, V0_steps)
alpha_range = np.linspace(alpha_min, alpha_max, alpha_steps)
