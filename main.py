# run_dynamic.py

import numpy as np
import matplotlib.pyplot as plt

from mesh       import generate_1d_bar_mesh
from assembly   import assemble_global_stiffness
from mass       import assemble_global_mass
from damping    import assemble_rayleigh_damping
from solver import newmark_beta

# --- Problem parameters ---
L       = 1.0
n_elems = 20
E       = 210e9
A       = 1e-4
rho     = 7850
alpha   = 0.02
beta_r  = 0.001

# Time integration
dt      = 1e-4
t_max   = 0.05

# 1) Mesh & matrices
nodes, elements, lengths = generate_1d_bar_mesh(L, n_elems)
K = assemble_global_stiffness(nodes, elements, E, A)
M = assemble_global_mass(nodes, elements, lengths, rho, A, lumped=False)
C = assemble_rayleigh_damping(M, K, alpha, beta_r)

# 2) Initial conditions
N = nodes.size
u0 = np.zeros(N)
v0 = np.zeros(N)

# 3) Define load: step load F_end applied at last node at t=0
def F_func(t):
    F = np.zeros(N)
    F[-1] = 1000.0  # constant for all t ≥ 0
    return F

# 4) Run Newmark-Beta
t, U, V, A_hist = newmark_beta(M, C, K, F_func, u0, v0, dt, t_max)

# 5) Plot displacement at free end over time
plt.figure()
plt.plot(t, U[:, -1])
plt.xlabel('Time [s]')
plt.ylabel('Displacement at node N [m]')
plt.title('Dynamic response (Newmark-Beta)')
plt.grid(True)
plt.show()
