import numpy as np
import scipy.sparse as sp
from typing import Tuple

def elasticity_matrix(E: float, nu: float) -> np.ndarray:
    """
    Return the 6×6 constitutive matrix D for isotropic linear elasticity.
    """
    # Lame parameters
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    # Construct D
    D = np.zeros((6, 6))
    D[0:3, 0:3] = lam
    D[np.diag_indices(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    return D

def element_stiffness(coords: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the 12×12 stiffness matrix for a linear tetrahedral element.

    Parameters
    ----------
    coords : (4,3) array of node coordinates for the element
    D      : (6,6) elasticity matrix

    Returns
    -------
    Ke : (12,12) element stiffness matrix
    V  : float, element volume
    """
    # Build matrix X for shape-function coefficients
    X = np.ones((4, 4))
    X[:, 1:] = coords
    detX = np.linalg.det(X)
    V = abs(detX) / 6.0

    # Inverse to get gradients of shape functions
    invX = np.linalg.inv(X)
    grads = invX[:, 1:]  # each row i: [dNi/dx, dNi/dy, dNi/dz]

    # Assemble B matrix (6×12)
    B = np.zeros((6, 12))
    for i in range(4):
        bi, ci, di = grads[i]
        idx = 3 * i
        # normal strains
        B[0, idx]     = bi
        B[1, idx + 1] = ci
        B[2, idx + 2] = di
        # shear strains
        B[3, idx]     = ci; B[3, idx + 1] = bi
        B[4, idx + 1] = di; B[4, idx + 2] = ci
        B[5, idx]     = di; B[5, idx + 2] = bi

    # Element stiffness
    Ke = B.T @ D @ B * V
    return Ke, V

def assemble_global_stiffness(nodes: np.ndarray,
                              elems: np.ndarray,
                              E: float,
                              nu: float) -> sp.csr_matrix:
    """
    Assemble the global stiffness matrix for a 3D tetrahedral mesh.

    Parameters
    ----------
    nodes : (n_nodes,3) array of node coordinates
    elems : (n_elems,4) array of tetra connectivity (node indices)
    E     : Young's modulus
    nu    : Poisson's ratio

    Returns
    -------
    K : (3n × 3n) sparse CSR stiffness matrix
    """
    n_nodes = nodes.shape[0]
    ndof = 3 * n_nodes
    D = elasticity_matrix(E, nu)

    # Preallocate lists for COO format
    rows, cols, data = [], [], []

    # Loop over elements
    for elem in elems:
        coords = nodes[elem]
        Ke, _ = element_stiffness(coords, D)
        # Scatter into global
        for i_local, ni in enumerate(elem):
            for j_local, nj in enumerate(elem):
                for a in range(3):
                    for b in range(3):
                        rows.append(3*ni + a)
                        cols.append(3*nj + b)
                        data.append(Ke[3*i_local + a, 3*j_local + b])

    # Build sparse matrix
    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof))
    return K.tocsr()

if __name__ == "__main__":
    # Example usage
    import scipy.sparse as _sp
    nodes = np.load("nodes.npy")
    elems = np.load("elems.npy")
    E = 210e9
    nu = 0.3
    K = assemble_global_stiffness(nodes, elems, E, nu)
    _sp.save_npz("K.npz", K)
    print(f"Assembled global stiffness for {nodes.shape[0]} nodes; saved to K.npz")
