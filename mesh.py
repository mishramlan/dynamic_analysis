import numpy as np
from typing import Tuple, List

def generate_1d_bar_mesh(length: float,
                         n_elems: int
                        ) -> Tuple[np.ndarray, 
                                   List[Tuple[int,int]], 
                                   np.ndarray]:
    """
    Generate nodes, element connectivity, and element lengths
    for a 1D bar.

    Parameters
    ----------
    length : float
        Total length of the bar.
    n_elems : int
        Number of equal-length elements.

    Returns
    -------
    nodes : np.ndarray, shape (n_elems+1,)
        Coordinates of each node along the bar [0, length].
    elements : list of (int,int)
        Connectivity pairs (node_i, node_j) for each element.
    lengths : np.ndarray, shape (n_elems,)
        Length of each element.
    """
    # 1) Node coordinates
    nodes = np.linspace(0.0, length, n_elems + 1)

    # 2) Element connectivity
    elements = [(i, i + 1) for i in range(n_elems)]

    # 3) Element lengths
    lengths = np.diff(nodes)  # consecutive differences

    return nodes, elements, lengths