import numpy as np
import scipy.sparse as sp
from typing import Tuple


def element_mass_consistent(rho: float,
                             coords: np.ndarray
                            ) -> Tuple[np.ndarray, float]:
    """
    Consistent 12×12 mass matrix for a linear tetrahedral element:
      m_e = (rho*V/20) * [C_ij ⊗ I3]  where C_ii=2, C_ij=1 for i≠j,
    and V = det([1 x y z]) / 6.

    Parameters
    ----------
    rho    : density
    coords : (4,3) array of node coords for this element

    Returns
    -------
    m_e : (12×12) element mass matrix
    V   : element volume
    """
    # Compute volume via determinant
    X = np.ones((4, 4))
    X[:, 1:] = coords
    detX = np.linalg.det(X)
    V = abs(detX) / 6.0

    # Coefficient matrix C (4×4): 2 on diag, 1 off-diag
    C = np.ones((4, 4))
    np.fill_diagonal(C, 2.0)

    # Build m_e = (rho*V/20) * (C ⊗ I3)
    factor = rho * V / 20.0
    m_e = np.zeros((12, 12))
    for i in range(4):
        for j in range(4):
            block = factor * C[i, j] * np.eye(3)
            m_e[3*i:3*i+3, 3*j:3*j+3] = block

    return m_e, V


def element_mass_lumped(rho: float,
                         coords: np.ndarray
                        ) -> Tuple[np.ndarray, float]:
    """
    Lumped mass: sum consistent mass rows into diagonal entries.
    """
    m_consistent, V = element_mass_consistent(rho, coords)
    row_sums = np.sum(m_consistent, axis=1)
    m_e = np.diag(row_sums)
    return m_e, V


def assemble_global_mass(nodes: np.ndarray,
                         elems: np.ndarray,
                         rho: float,
                         lumped: bool = False
                        ) -> sp.csr_matrix:
    """
    Assemble global mass matrix M for a 3D tetrahedral mesh.

    Returns a sparse (3n×3n) CSR matrix.
    """
    n_nodes = nodes.shape[0]
    ndof = 3 * n_nodes
    rows, cols, data = [], [], []

    for elem in elems:
        coords = nodes[elem]
        if lumped:
            m_e, _ = element_mass_lumped(rho, coords)
        else:
            m_e, _ = element_mass_consistent(rho, coords)

        for i_local, ni in enumerate(elem):
            for j_local, nj in enumerate(elem):
                for a in range(3):
                    for b in range(3):
                        rows.append(3*ni + a)
                        cols.append(3*nj + b)
                        data.append(m_e[3*i_local + a, 3*j_local + b])

    M = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof))
    return M.tocsr()
