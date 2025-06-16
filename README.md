# Dynamic FEA with Python

This repository contains:
- 1D and 3D FEA solvers (static & dynamic)
- Newmark‐Beta implicit solver (`dynamic_solver.py`)
- Explicit central‐difference visco‐plastic solver (`dynamic_solver_explicit.py`)
- Mesh import & preprocessing (`import_mesh.py`, `mesh_reader.py`)
- Example run scripts (`run_dynamic_3d.py`, `run_explicit_dynamic_3d.py`)
- Dependencies in `requirements.txt`

## Getting Started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
