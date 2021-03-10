#!/usr/bin/env python
import numpy as np
import argparse
import logging
import os

import atooms
import atooms.trajectory
import gsd.hoomd

from my_hoomd_utils import calculate_msd, calculate_radial_distribution_function, calculate_structure_factor, calculate_self_intermediate_scattering_function, calculate_overlap, calculate_dynamic_susceptiblility, default_tgrid
import model_utils
from plotting_functions import *


p = argparse.ArgumentParser()
p.add_argument("--num_partitions", type=int, default=1)

p.add_argument("--traj_file", type=str)

p.add_argument("--msd", action='store_true')
p.add_argument("--gr", action='store_true')
p.add_argument("--Sk", action='store_true')
p.add_argument("--Fs", action='store_true')
p.add_argument("--Q", action='store_true')
p.add_argument("--chi4_Qs", action='store_true')
p.add_argument("--chi4_Fs", action='store_true')
p.add_argument("--tmax", type=float)
p.add_argument("--T", type=float)
p.add_argument("--rho", type=float)
p.add_argument("--model_name", type=str)
args = p.parse_args()

num_partitions = args.num_partitions
traj_file = args.traj_file
tmax = args.tmax



data_folder = os.path.join( os.path.split(os.path.dirname(traj_file))[0], "log")
base_name = os.path.splitext(os.path.basename(traj_file))[0]
out_file_msd = "%s/%s_msd.dat" % (data_folder, base_name)
out_file_msd_qty = "%s/%s_msd_qty.dat" % (data_folder, base_name)
out_file_gr = "%s/%s_gr.dat" % (data_folder, base_name)
out_file_Sk = "%s/%s_Sk.dat" % (data_folder, base_name)
out_file_Fs = "%s/%s_Fs.dat" % (data_folder, base_name)
out_file_Fs_qty = "%s/%s_Fs_qty.dat" % (data_folder, base_name)
out_file_Q = "%s/%s_Q.dat" % (data_folder, base_name)
out_file_Q_qty = "%s/%s_Q_qty.dat" % (data_folder, base_name)
out_file_chi4_Qs = "%s/%s_chi4_Qs.dat" % (data_folder, base_name)
out_file_chi4_Fs = "%s/%s_chi4_Fs.dat" % (data_folder, base_name)

state_file = "%s/state.dat" % (data_folder)

# Get rho, T from state file.
if os.path.isfile(state_file):
    with open(state_file, "r") as f:
        lines = [line.rstrip() for line in f]
        data = lines[1].split(", ")

        model_name = str(data[0])
        T = float(data[1])
        tau_thermostat = float(data[2])
        rho = float(data[3])
        npart = int(data[4])
        dt = float(data[5])
# Alternatively, supply T and rho yourself.
else:
    T = args.T
    rho = args.rho
    model_name = args.model_name

model = model_utils.models[model_name](T=T, rho=rho)


if args.msd:
    print("Starting with MSD")
    ts, msds = calculate_msd(traj_file, num_partitions=num_partitions, out_file=out_file_msd, out_file_quantities=out_file_msd_qty)

if args.gr:
    print("Starting with g(r)")
    calculate_radial_distribution_function(traj_file, out_file=out_file_gr)

if args.Sk:
    print("Starting with Structure Factor")
    ks = calculate_structure_factor(traj_file, out_file=out_file_Sk, ksamples=60, nk=32)

if args.Fs:
    print("Starting with Fs")

    q_values =  model.get_qmax()

    calculate_self_intermediate_scattering_function(traj_file, q_values, out_file=out_file_Fs, out_file_quantities=out_file_Fs_qty, nk=15, dk=0.03, fix_cm=True)

if args.Q:
    print("Starting with overlap function.")

    a = 0.3

    calculate_overlap(traj_file, a, out_file=out_file_Q, out_file_quantities=out_file_Q_qty)


if args.chi4_Qs:
    print("Starting with dynamic susceptibility of self overlap.")

    a = 0.3
    with atooms.trajectory.Trajectory(traj_file) as traj:
        steps = traj.steps
    
    tgrid = default_tgrid(steps, t_upper_limit=tmax)

    calculate_dynamic_susceptiblility(traj_file, out_file=out_file_chi4_Qs, corr="self_overlap", a=a, tgrid=tgrid)#, out_file_quantities=out_file_Q_qty)

if args.chi4_Fs:
    print("Starting with dynamic susceptibility of self intermediate scattering function.")

    q_values =  model.get_qmax()

    with atooms.trajectory.Trajectory(traj_file) as traj:
        steps = traj.steps
    
    tgrid = default_tgrid(steps, t_upper_limit=tmax)

    calculate_dynamic_susceptiblility(traj_file, out_file=out_file_chi4_Fs, corr="self_intermediate_scattering", nk=8, dk=0.1, fix_cm=False, kgrid=q_values, tgrid=tgrid)#, out_file_quantities=out_file_Q_qty)
    

