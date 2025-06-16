import numpy as np
import matplotlib.pyplot as plt

from explicit_solver import explicit_central_difference


def main():
    # --- Load mesh data ---
    nodes = np.load("nodes.npy")    # (n_nodes, 3)
    elems = np.load("elems.npy")    # (n_elems, 4)

    # --- Boundary selection ---
    tol = 1e-8
    fixed_nodes = np.where(np.abs(nodes[:,1] - 0.0) < tol)[0]
    fixed_dofs = np.hstack([3*fixed_nodes + d for d in (0,1,2)])

    L = nodes[:,1].max()
    loaded_nodes = np.where(np.abs(nodes[:,1] - L) < tol)[0]
    loaded_dofs = 3*loaded_nodes + 1  # Y-direction

    # --- Material & damping parameters ---
    E        = 210e9      # Young's modulus [Pa]
    nu       = 0.3        # Poisson's ratio
    rho      = 7850.0     # density [kg/m^3]
    alpha    = 0.02       # Rayleigh mass-proportional damping
    beta     = 0.001      # Rayleigh stiffness-proportional damping
    sigma_y  = 250e6      # yield stress [Pa]
    eta      = 1e9        # viscosity parameter

    # --- Time integration settings ---
    dt     = 1e-4
    t_max  = 0.05

    # --- Sinusoidal load parameters ---
    amplitude = 1000.0    # peak load [N]
    frequency = 50.0      # Hz

    # Define force function full-domain
    def F_func(t):
        F = np.zeros(3 * nodes.shape[0])
        F_val = amplitude * np.sin(2*np.pi*frequency * t)
        if t >= 0:
            F[loaded_dofs] = F_val / loaded_dofs.size
        # Zero out fixed dofs
        F[fixed_dofs] = 0.0
        return F

    # --- Run explicit central-difference solver ---
    print("Running explicit visco-plastic dynamic solver...")
    t, U_hist, V_hist, stress_vm_hist = explicit_central_difference(
        nodes, elems,
        E, nu, rho,
        alpha, beta,
        F_func,
        fixed_dofs, loaded_dofs,
        dt, t_max,
        sigma_y, eta
    )

    # --- Post-processing ---
    # Displacement: average Y-displacement on loaded nodes
    disp_loaded = np.mean([U_hist[:, dof] for dof in loaded_dofs], axis=0)

    # Plot results
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    axes[0].plot(t, disp_loaded)
    axes[0].set_ylabel('Displacement [m]')
    axes[0].set_title('Avg Y-Displacement on Loaded Face')
    axes[0].grid(True)

    axes[1].plot(t, stress_vm_hist)
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Max Von Mises Stress [Pa]')
    axes[1].set_title('Visco-Plastic Stress Response')
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
