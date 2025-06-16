import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Callable, Tuple, Union, List



def apply_boundary_conditions(M, C, K, fixed_dofs):
    """
    Reduce M, C, K by eliminating fixed_dofs.
    Returns reduced matrices and free_dofs array.
    """
    ndof = M.shape[0]
    all_dofs = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # slice sparse or dense
    def reduce_matrix(A):
        return A[free_dofs, :][:, free_dofs]

    M_red = reduce_matrix(M)
    C_red = reduce_matrix(C)
    K_red = reduce_matrix(K)
    return M_red, C_red, K_red, free_dofs




def newmark_beta(M: Union[np.ndarray, sp.spmatrix],
                 C: Union[np.ndarray, sp.spmatrix],
                 K: Union[np.ndarray, sp.spmatrix],
                 F_func: Callable[[float], np.ndarray],
                 u0: np.ndarray,
                 v0: np.ndarray,
                 dt: float,
                 t_max: float,
                 beta: float = 1/4,
                 gamma: float = 1/2
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Newmark-Beta time integration for the dynamic system:
        M * a + C * v + K * u = F(t)

    Supports both dense (numpy) and sparse (scipy.sparse) matrices.

    Parameters
    ----------
    M : (N×N) array or sparse matrix
        Global mass matrix (possibly reduced by BCs).
    C : (N×N) array or sparse matrix
        Global damping matrix.
    K : (N×N) array or sparse matrix
        Global stiffness matrix.
    F_func : function(t) -> ndarray, shape (N,)
        Time-dependent force vector generator.
    u0 : ndarray, shape (N,)
        Initial displacement vector.
    v0 : ndarray, shape (N,)
        Initial velocity vector.
    dt : float
        Time step size.
    t_max : float
        Total simulation time.
    beta : float
        Newmark-Beta parameter (default 1/4).
    gamma : float
        Newmark-Beta parameter (default 1/2).

    Returns
    -------
    t : ndarray, shape (n_steps+1,)
        Time array.
    U : ndarray, shape (n_steps+1, N)
        Displacement history.
    V : ndarray, shape (n_steps+1, N)
        Velocity history.
    A : ndarray, shape (n_steps+1, N)
        Acceleration history.
    """
    # Number of steps
    n_steps = int(np.ceil(t_max / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    N = u0.size

    # Allocate history arrays
    U = np.zeros((n_steps + 1, N))
    V = np.zeros((n_steps + 1, N))
    A = np.zeros((n_steps + 1, N))

    # Set initial conditions
    U[0, :] = u0
    V[0, :] = v0

    # Initial acceleration: M * a0 = F(0) - C*v0 - K*u0
    R0 = F_func(0.0) - (C.dot(v0) if hasattr(C, 'dot') else C @ v0) - (K.dot(u0) if hasattr(K, 'dot') else K @ u0)
    if sp.issparse(M):
        A[0, :] = spla.spsolve(M, R0)
    else:
        A[0, :] = np.linalg.solve(M, R0)

    # Effective stiffness matrix K_eff
    K_eff = (K
             + (gamma / (beta * dt)) * C
             + (1.0 / (beta * dt**2)) * M)

    # Time-stepping loop
    for i in range(n_steps):
        u_n = U[i, :]
        v_n = V[i, :]
        a_n = A[i, :]
        t_np1 = t[i + 1]

        # Predictors
        u_pred = u_n + dt * v_n + dt**2 * (0.5 - beta) * a_n
        v_pred = v_n + dt * (1.0 - gamma) * a_n

        # Effective force
        F_np1 = F_func(t_np1)
        term_M = M.dot(u_pred) * (1.0 / (beta * dt**2)) if hasattr(M, 'dot') else M @ u_pred * (1.0 / (beta * dt**2))
        term_C = C.dot(u_pred) * (gamma / (beta * dt)) if hasattr(C, 'dot') else C @ u_pred * (gamma / (beta * dt))
        F_eff = F_np1 + term_M + term_C

        # Solve for new displacement
        if sp.issparse(K_eff):
            u_np1 = spla.spsolve(K_eff, F_eff)
        else:
            u_np1 = np.linalg.solve(K_eff, F_eff)

        # Corrector: velocities and accelerations
        a_np1 = (1.0 / (beta * dt**2)) * (u_np1 - u_pred)
        v_np1 = v_pred + gamma * dt * a_np1

        # Store
        U[i + 1, :] = u_np1
        V[i + 1, :] = v_np1
        A[i + 1, :] = a_np1

    return t, U, V, A
