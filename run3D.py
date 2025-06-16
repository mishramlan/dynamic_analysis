import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from assembly3d import assemble_global_stiffness, elasticity_matrix
from mass3d import assemble_global_mass
from damping import assemble_rayleigh_damping
from solver import newmark_beta, apply_boundary_conditions


def compute_element_stress(coords, D, u_full, elem):
    """
    Compute 6-component stress for one tetrahedral element.
    coords: (4,3), D: (6,6), u_full: (ndof,), elem: node indices
    Returns stress vector sigma of length 6.
    """
    # Build X matrix for shape gradients
    X = np.ones((4,4))
    X[:,1:] = coords
    invX = np.linalg.inv(X)
    grads = invX[:,1:]  # (4,3) rows: grad of N_i

    # Build B matrix (6x12)
    B = np.zeros((6,12))
    for i_local in range(4):
        bi, ci, di = grads[i_local]
        idx = 3 * i_local
        B[0, idx    ] = bi
        B[1, idx + 1] = ci
        B[2, idx + 2] = di
        B[3, idx    ] = ci; B[3, idx + 1] = bi
        B[4, idx + 1] = di; B[4, idx + 2] = ci
        B[5, idx    ] = di; B[5, idx + 2] = bi

    # Extract local displacement u_e
    u_e = np.zeros(12)
    for i_local, ni in enumerate(elem):
        u_e[3*i_local:3*i_local+3] = u_full[3*ni:3*ni+3]

    sigma = D.dot(B.dot(u_e))
    return sigma


def von_mises(sigma):
    """
    Compute Von Mises equivalent stress from 6-component stress.
    sigma: [s11, s22, s33, s12, s23, s13]
    """
    s11, s22, s33, s12, s23, s13 = sigma
    return np.sqrt(0.5*((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 + 6*(s12**2 + s23**2 + s13**2)))


def main():
    # --- Load mesh data ---
    nodes = np.load("nodes.npy")    # (n_nodes, 3)
    elems = np.load("elems.npy")    # (n_elems, 4)

    # --- Material & damping parameters ---
    E     = 210e9       # Young's modulus [Pa]
    nu    = 0.3         # Poisson's ratio
    rho   = 7850.0      # density [kg/m^3]
    alpha = 0.02        # Rayleigh mass-proportional damping
    beta  = 0.001       # Rayleigh stiffness-proportional damping

    # --- Assemble global matrices ---
    print("Assembling global stiffness K...")
    K = assemble_global_stiffness(nodes, elems, E, nu)
    print("Assembling global mass M...")
    M = assemble_global_mass(nodes, elems, rho, lumped=False)
    print("Assembling Rayleigh damping C...")
    C = assemble_rayleigh_damping(M, K, alpha, beta)

    # --- Boundary selection ---
    tol = 1e-8
    fixed_nodes = np.where(np.abs(nodes[:,1] - 0.0) < tol)[0]
    fixed_dofs = np.hstack([3*fixed_nodes + d for d in (0,1,2)])
    L = nodes[:,1].max()
    loaded_nodes = np.where(np.abs(nodes[:,1] - L) < tol)[0]
    loaded_dofs = 3*loaded_nodes + 1  # Y-direction
    loaded_elems = [i for i, elem in enumerate(elems) if any(n in loaded_nodes for n in elem)]

    # --- Reduce system ---
    M_red, C_red, K_red, free_dofs = apply_boundary_conditions(M, C, K, fixed_dofs)

    # --- Sinusoidal load parameters ---
    amplitude = 5000.0    # peak load [N]
    frequency = 50.0      # frequency [Hz]
    omega = 2 * np.pi * frequency

    # --- Force generators ---
    def F_full(t):
        F = np.zeros(M.shape[0])
        F_val = amplitude * np.sin(omega * t)
        F[loaded_dofs] = F_val / loaded_dofs.size
        return F
    def F_func(t):
        return F_full(t)[free_dofs]

    # --- Initial conditions ---
    n_red = M_red.shape[0]
    u0 = np.zeros(n_red)
    v0 = np.zeros(n_red)

    # --- Time integration settings ---
    dt    = 1e-4   # time step [s]
    t_max = 0.2   # total run time [s]

    # --- Run Newmark-Beta ---
    print("Running Newmark-Beta dynamic analysis...")
    t, U_red, V_red, A_red = newmark_beta(M_red, C_red, K_red, F_func, u0, v0, dt, t_max)

    # --- Scatter back to full DOFs ---
    n_steps = U_red.shape[0]
    ndof = M.shape[0]
    U_full = np.zeros((n_steps, ndof))
    U_full[:, free_dofs] = U_red

    # --- Compute stresses at loaded elements over time ---
    D = elasticity_matrix(E, nu)
    stress_vm = np.zeros(n_steps)
    for i in range(n_steps):
        sigma_vm_vals = []
        u_vec = U_full[i]
        for idx in loaded_elems:
            elem = elems[idx]
            coords = nodes[elem]
            sigma = compute_element_stress(coords, D, u_vec, elem)
            sigma_vm_vals.append(von_mises(sigma))
        stress_vm[i] = np.max(sigma_vm_vals)

    # --- Post-processing plots ---
    # Gather displacement and load histories
    disp_loaded = U_full[:, loaded_dofs].mean(axis=1)
    loads = np.array([F_full(ti)[loaded_dofs].sum() for ti in t])

    # Plot in three subplots
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

    axes[0].plot(t, loads)
    axes[0].set_ylabel('Load [N]')
    axes[0].set_title('Sinusoidal Load')
    axes[0].grid(True)

    axes[1].plot(t, disp_loaded)
    axes[1].set_ylabel('Displacement [m]')
    axes[1].set_title('Avg Y-Displacement')
    axes[1].grid(True)

    axes[2].plot(t, stress_vm)
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Von Mises Stress [Pa]')
    axes[2].set_title('Max Von Mises Stress on Loaded Region')
    axes[2].grid(True)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
