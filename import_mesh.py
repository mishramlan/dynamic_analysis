# import_mesh.py

import gmsh
import sys

def main(step_file: str = "/Users/User/Documents/dynamic_fea/Traverse.step", msh_file: str = "/Users/User/Documents/dynamic_fea/Traverse.msh"):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)     # print messages
    gmsh.model.add("model")

    # 1) Import the STEP geometry
    gmsh.merge(step_file)

    # 2) Synchronize CLI <-> CAD kernel
    gmsh.model.occ.synchronize()

    # 3) Generate a 3D mesh (tetrahedra)
    gmsh.model.mesh.generate(3)

    # 4) Write to disk
    gmsh.write(msh_file)
    print(f"Mesh written to {msh_file}")

    gmsh.finalize()

if __name__ == "__main__":
    # usage: python import_mesh.py [part.step] [mesh.msh]
    args = sys.argv[1:]
    main(*args)