import numpy as np

def assemble_rayleigh_damping(M: np.ndarray,
                              K: np.ndarray,
                              alpha: float,
                              beta: float
                             ) -> np.ndarray:
    """
    Assemble the global Rayleigh damping matrix.

    Parameters
    ----------
    M : (N×N) array
        Global mass matrix.
    K : (N×N) array
        Global stiffness matrix.
    alpha : float
        Mass-proportional damping coefficient.
    beta : float
        Stiffness-proportional damping coefficient.

    Returns
    -------
    C : (N×N) array
        Global damping matrix C = alpha*M + beta*K.
    """
    return alpha * M + beta * K