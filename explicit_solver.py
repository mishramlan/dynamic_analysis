import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Callable, Tuple

from constitutive_laws import update_viscoplastic, von_mises_eq
from assembly3d import elasticity_matrix
from mass3d import assemble_global_mass
from damping import assemble_rayleigh_damping


def explicit_central_difference(
    nodes: np.ndarray,
    elems: np.ndarray,
    E: float,
    nu: float,
    rho: float,
    alpha: float,
    beta: float,
    F_func: Callable[[float], np.ndarray],
    fixed_dofs: np.ndarray,
    loaded_dofs: np.ndarray,
    dt: float,
    t_max: float,
    sigma_y: float,
    eta: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Explicit central-difference time integration with visco-plasticity.

    Returns: t, U_full, V_full, stress_vm (max von Mises on loaded elems)
    """
    n_nodes = nodes.shape[0]
    ndof = 3 * n_nodes
    n_elems = elems.shape[0]

    # Lumped mass for explicit scheme
    M = assemble_global_mass(nodes, elems, rho, lumped=True)
    # Convert to diagonal inverse
    if sp.issparse(M):
        M_diag = M.diagonal()
    else:
        M_diag = np.diag(M)
    M_inv = 1.0 / M_diag

    # Global damping (lumped mass + stiffness) - optional
    K_dummy = assemble_global_mass(nodes, elems, rho, lumped=True)  # placeholder
    C = assemble_rayleigh_damping(M, K_dummy, alpha, beta)

    # Precompute constitutive matrix
    D = elasticity_matrix(E, nu)

    # Precompute B matrices and volumes
    B_mats = []  # list of (6,12)
    vols = []    # volumes
    for elem in elems:
        coords = nodes[elem]
        X = np.ones((4,4))
        X[:,1:] = coords
        detX = np.linalg.det(X)
        V = abs(detX) / 6.0
        invX = np.linalg.inv(X)
        grads = invX[:,1:]
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
        B_mats.append(B)
        vols.append(V)

    # Initialize state
    u = np.zeros(ndof)
    v = np.zeros(ndof)
    a = np.zeros(ndof)
    eps_p = np.zeros((n_elems, 6))  # plastic strain per element

    # Time setup
    n_steps = int(np.ceil(t_max / dt))
    t = np.linspace(0, n_steps*dt, n_steps+1)

    # Histories
    U_hist = np.zeros((n_steps+1, ndof))
    V_hist = np.zeros((n_steps+1, ndof))
    stress_vm_hist = np.zeros(n_steps+1)

    # Initial internal force f_int and acceleration
    f_int = np.zeros(ndof)
    # assume zero initial f_int, so a0 = M^{-1}(F0 - C v0)
    F0 = F_func(0)
    a = M_inv * (F0 - C.dot(v))

    # Identify loaded element indices
    loaded_elems = [i for i, elem in enumerate(elems) if any(n in np.unique(loaded_dofs//3) for n in elem)]

    # Time integration
    for i in range(n_steps):
        # Store history
        U_hist[i, :] = u
        V_hist[i, :] = v
        # Compute stress max on loaded region
        sigma_vm_vals = []
        for idx in loaded_elems:
            elem = elems[idx]
            # compute sigma for element
            coords = nodes[elem]
            # local disp
            u_e = np.concatenate([u[3*ni:3*ni+3] for ni in elem])
            # compute total strain: eps = B u_e
            B = B_mats[idx]
            eps = B.dot(u_e)
            sigma, eps_p[idx] = update_viscoplastic(eps_new=eps, eps_p_old=eps_p[idx], D=D, sigma_y=sigma_y, eta=eta, dt=dt)
            sigma_vm_vals.append(von_mises_eq(sigma))
        stress_vm_hist[i] = max(sigma_vm_vals) if sigma_vm_vals else 0.0

        # 1) Predict displacement and velocity
        u_np1 = u + dt*v + 0.5*dt*dt*a
        v_half = v + 0.5*dt*a

        # 2) Compute internal force at u_np1
        f_int[:] = 0
        for idx, elem in enumerate(elems):
            B = B_mats[idx]
            coords = nodes[elem]
            # local disp
            u_e = np.concatenate([u_np1[3*ni:3*ni+3] for ni in elem])
            eps = B.dot(u_e)
            sigma, eps_p[idx] = update_viscoplastic(eps_new=eps, eps_p_old=eps_p[idx], D=D, sigma_y=sigma_y, eta=eta, dt=dt)
            # element internal force
            fe = B.T.dot(sigma) * vols[idx]
            # scatter
            for j_local, nj in enumerate(elem):
                f_int[3*nj:3*nj+3] += fe[3*j_local:3*j_local+3]

        # 3) Update acceleration
        F_np1 = F_func(t[i+1])
        a_np1 = M_inv * (F_np1 - f_int - C.dot(v_half))

        # 4) Update velocity
        v_np1 = v_half + 0.5*dt*a_np1

        # Advance
        u, v, a = u_np1, v_np1, a_np1

    # store final state
    U_hist[-1, :] = u
    V_hist[-1, :] = v
    stress_vm_hist[-1] = stress_vm_hist[-2]

    return t, U_hist, V_hist, stress_vm_hist



