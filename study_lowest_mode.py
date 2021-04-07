#!/usr/bin/env python
import os
num_ppn = str(1)
os.environ["OMP_NUM_THREADS"] = num_ppn # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = num_ppn
os.environ["MKL_NUM_THREADS"] = num_ppn
os.environ["VECLIB_MAXIMUM_THREADS"] = num_ppn
os.environ["NUMEXPR_NUM_THREADS"] = num_ppn

from lerner_group_base_imports import *
from mpl_toolkits.mplot3d import Axes3D
from softspot.softspot import SoftSpot
import argparse
import numpy as np
from plotting_functions import *

# from lerner_group.visualization_tools_3d import render_snapshot_3d

p = argparse.ArgumentParser()
p.add_argument("--solid_file", type=str)
p.add_argument("--model", type=str)
args = p.parse_args()


model_name = args.model
solid_file = args.solid_file


snap, _ = read_trajectory(solid_file, frame=0)

pos = snap['pos']
ptypes = snap['ptypes']
npart = snap['npart']
diam = np.zeros(npart)
box_size = snap['box']
ndim = box_size.shape[0]

if model_name == "force_shifted_lj":
    sim = InitializeModel(model_name="thomas", r=pos, diameter=ptypes, box_size=box_size, strain=0.0).sim
    radius = 0.5*np.ones(npart)
elif model_name == "sticky_spheres":
    sim = InitializeModel(model_name="sticky_spheres", r=pos, diameter=ptypes, box_size=box_size, strain=0.0, options={'rcut':1.2}).sim

sim.pairwise.compute_everything()
print("typGrad = %g, typContactForce = %g, ratio = %g,  pressure = %g, rho = %g" % (sim.system.typical_grad, sim.pairwise.typical_contact_force, sim.system.typical_grad / sim.pairwise.typical_contact_force, sim.system.thermo_pre, npart/np.prod(sim.system.boxl)))


cg_options = dict(ftol=1e-7, x0=None, maxiter=int(1e6))
print("Calculating K...")
K, K_born, K_na = calculateBulkModulus(**cg_options)
print("Calculating G...")
G_born, G_na = calculateShearModulus(**cg_options)
G = G_born + G_na

print("G = %g, K = %g" % (G, K))

hessian = get_sparse_2b_hessian()
soft_spot = SoftSpot(ndim=ndim, npart=npart, hessian=hessian)

kappa_harmonic_all, psi_all, e_harmonic_all = diagonalize_hessian(n_modes=4, calc_participation=True)
kappa_harmonic = kappa_harmonic_all[-1]
e_harmonic = e_harmonic_all[-1]
psi = psi_all[:, -1]

result = soft_spot.find(psi, mode='cg')
pi = result['pi']
kappa = result['kappa']
e = participation_ratio(pi)
e_harmonic = participation_ratio(psi)

# Also find lowest mode from participation ratio mode.
tries = 10
kappa_participation_min = np.inf
for i in range(tries):
    random_guess = np.random.rand(npart*ndim)
    zeta, kappa_zeta = get_nonlinear_mode(random_guess, order='participation')
    result = soft_spot.find(zeta, mode='cg')
    kappa_participation = result['kappa']
    print("[%d] kappa zeta = %g, kappa participation mapped = %g" % (i, kappa_zeta, kappa_participation))
    if kappa_participation < kappa_participation_min:
        kappa_participation_min = kappa_participation
        pi_participation = result['pi']
        e_participation = participation_ratio(pi_participation)


print("kappa = %g, kappa_harmonic = %g, kappa_participation = %g, e = %g, e harmonic = %g, e_participation = %g" % (kappa, kappa_harmonic, kappa_participation_min, e, e_harmonic, e_participation))

ax1, ax2, ax3 = init_fig(grid=(1,3), projections=['3d', '3d', '3d'])


scale=10
remove_fraction = 0.95
options = {'linewidths': 1.0}

render_field_3d(ax1, pos, psi, length=scale, remove_fraction=remove_fraction, options=options, tick_labels=False)
render_field_3d(ax2, pos, pi, length=scale, remove_fraction=remove_fraction, options=options, tick_labels=False)
render_field_3d(ax3, pos, pi_participation, length=scale, remove_fraction=remove_fraction, options=options, tick_labels=False)

# cmap = sns.color_palette("muted")
# main_color = np.array(cmap[0] + (1,))

# colors = np.zeros((npart, 4))
# for i in range(npart):
#     colors[i, :] = main_color

# render_snapshot_3d(ax2, pos, radius, colors, outline_width=0.1, roughness=0.8, specular=0.2, width=400, height=400)


plt.tight_layout()
plt.show()



