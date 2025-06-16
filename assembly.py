import numpy as np
from typing import Tuple, List
from mesh import generate_1d_bar_mesh

def element_stiffness(E: float, A: float, length: float) -> np.ndarray:
    """
    Compute the 2×2 stiffness matrix for a single linear 1D element.
    
    k_e = (A*E / length) * [[ 1, -1],
                            [-1,  1]]
    """
    k = (A * E / length)
    return k * np.array([[1.0, -1.0],
                         [-1.0, 1.0]])

def assemble_global_stiffness(nodes: np.ndarray,
                              elements: List[Tuple[int,int]],
                              E: float,
                              A: float) -> np.ndarray:
    """
    Assemble the global stiffness matrix K for the entire mesh.

    Parameters
    ----------
    nodes : np.ndarray
        1D array of node coordinates.
    elements : list of (int,int)
        Connectivity pairs for each element.
    E : float
        Young's modulus.
    A : float
        Cross-sectional area.

    Returns
    -------
    K : np.ndarray
        (N × N) global stiffness matrix, where N = len(nodes).
    """
    N = nodes.size
    K = np.zeros((N, N))

    for elem in elements:
        i, j = elem
        L_e = nodes[j] - nodes[i]
        k_e = element_stiffness(E, A, L_e)
        
        # Scatter into global K
        K[np.ix_([i, j], [i, j])] += k_e

    return K