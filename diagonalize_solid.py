#!/usr/bin/env python
import os
num_ppn = str(1)
os.environ["OMP_NUM_THREADS"] = num_ppn # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = num_ppn
os.environ["MKL_NUM_THREADS"] = num_ppn
os.environ["VECLIB_MAXIMUM_THREADS"] = num_ppn
os.environ["NUMEXPR_NUM_THREADS"] = num_ppn

from lerner_group_base_imports import *
import argparse
import numpy as np


p = argparse.ArgumentParser()
p.add_argument("--solid_file", type=str)
p.add_argument("--model", type=str)
p.add_argument("--n_modes", type=int)
p.add_argument("--solid_name", type=str)
args = p.parse_args()


model_name = args.model
solid_file = args.solid_file
n_modes = args.n_modes
solid_name = args.solid_name

data_folder = os.path.dirname(solid_file)
modes_folder = os.path.join(data_folder, "../modes")
os.makedirs(modes_folder, exist_ok=True)

modes_file = os.path.join(modes_folder, "%s.dat" % solid_name)
# if os.path.isile(modes_file) and not force_overwrite:
#     print("%s already exists, skipping." % solid_name)
#     raise SystemExit

snap, _ = read_trajectory(solid_file, frame=0)

pos = snap['pos']
ptypes = snap['ptypes']
npart = snap['npart']
diam = np.zeros(npart)
box_size = snap['box']
ndim = box_size.shape[0]

if model_name == "force_shifted_lj":
    sim = InitializeModel(model_name="thomas", r=pos, diameter=ptypes, box_size=box_size, strain=0.0).sim
elif model_name == "sticky_spheres":
    sim = InitializeModel(model_name="sticky_spheres", r=pos, diameter=ptypes, box_size=box_size, strain=0.0, options={'rcut':1.2}).sim

sim.pairwise.compute_everything()
ratio = sim.system.typical_grad / sim.pairwise.typical_contact_force
print("typGrad / typContactForce = %g" % (ratio))

if ratio > 1e-5:
    error_file = "%s/error%d.dat" % (modes_folder, solid_name)
    with open(error_file, "w") as f:
        f.write("typical_grad/typical_contact_force=%.12g\n" % ratio)
    raise SystemExit

kappa, eigenvectors, e = diagonalize_hessian(n_modes=n_modes, tol=0, calc_participation=True, sigma=1e-6)
kappa = kappa[ndim:]
eigenvectors = eigenvectors[:, ndim:]
e = e[ndim:]

# Save
columns = np.column_stack((kappa, e))
np.savetxt(modes_file, columns, fmt="%.12g %.12g", header="columns=kappa,participation")

