# mesh_reader.py

import meshio
import numpy as np

def read_mesh(msh_file: str = "/Users/User/Documents/dynamic_fea/Traverse.msh"):
    mesh = meshio.read(msh_file)
    points = mesh.points               # (n_nodes × 3) array
    if "tetra" in mesh.cells_dict:
        elems = mesh.cells_dict["tetra"]
    else:
        raise RuntimeError("No tetrahedral cells found in mesh.")
    return points, elems

def main(msh_file: str = "/Users/User/Documents/dynamic_fea/Traverse.msh"):
    nodes, elems = read_mesh(msh_file)
    np.save("nodes.npy", nodes)
    np.save("elems.npy", elems)
    print(f"Saved {nodes.shape[0]} nodes → nodes.npy")
    print(f"Saved {elems.shape[0]} tets  → elems.npy")

if __name__ == "__main__":
    # usage: python mesh_reader.py [mesh.msh]
    import sys
    args = sys.argv[1:]
    main(*args)