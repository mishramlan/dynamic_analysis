import numpy as np
from typing import Tuple, List
from mesh import generate_1d_bar_mesh

def element_mass_consistent(rho: float, A: float, L_e: float) -> np.ndarray:
    """
    Consistent 2×2 mass matrix for a 1D linear element:
      m_e = (rho*A*L_e/6) * [[2, 1],
                              [1, 2]]
    """
    factor = rho * A * L_e / 6.0
    return factor * np.array([[2.0, 1.0],
                              [1.0, 2.0]])

def element_mass_lumped(rho: float, A: float, L_e: float) -> np.ndarray:
    """
    Lumped (diagonal) mass matrix for a 1D element:
      m_e = (rho*A*L_e/2) * [[1, 0],
                             [0, 1]]
    """
    factor = rho * A * L_e / 2.0
    return factor * np.eye(2)

def assemble_global_mass(nodes: np.ndarray,
                         elements: List[Tuple[int,int]],
                         lengths: np.ndarray,
                         rho: float,
                         A: float,
                         lumped: bool = False
                        ) -> np.ndarray:
    """
    Assemble the N×N global mass matrix.

    Parameters
    ----------
    nodes : array of node coords
    elements : list of (i,j) node indices
    lengths : array of element lengths
    rho : material density
    A : cross-sectional area
    lumped : if True, use lumped mass; else consistent mass

    Returns
    -------
    M : np.ndarray, shape (N,N)
    """
    N = nodes.size
    M = np.zeros((N, N))

    for idx, (i, j) in enumerate(elements):
        L_e = lengths[idx]
        if lumped:
            m_e = element_mass_lumped(rho, A, L_e)
        else:
            m_e = element_mass_consistent(rho, A, L_e)

        # scatter into global matrix
        M[np.ix_([i, j], [i, j])] += m_e

    return M