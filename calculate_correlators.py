#!/usr/bin/env python
import numpy as np
import argparse
import logging
import os

import atooms
import atooms.trajectory
import gsd.hoomd

from my_hoomd_utils import calculate_msd, calculate_radial_distribution_function, calculate_structure_factor, calculate_self_intermediate_scattering_function
import model_utils
from plotting_functions import *


p = argparse.ArgumentParser()
p.add_argument("--num_partitions", type=int, default=1)

p.add_argument("--traj_file", type=str)

p.add_argument("--msd", action='store_true')
p.add_argument("--gr", action='store_true')
p.add_argument("--Sk", action='store_true')
p.add_argument("--Fs", action='store_true')
args = p.parse_args()

num_partitions = args.num_partitions
traj_file = args.traj_file



data_folder = os.path.join( os.path.split(os.path.dirname(traj_file))[0], "log")
base_name = os.path.splitext(os.path.basename(traj_file))[0]
out_file_msd = "%s/%s_msd.dat" % (data_folder, base_name)
out_file_msd_qty = "%s/%s_msd_qty.dat" % (data_folder, base_name)
out_file_gr = "%s/%s_gr.dat" % (data_folder, base_name)
out_file_Sk = "%s/%s_Sk.dat" % (data_folder, base_name)
out_file_Fs = "%s/%s_Fs.dat" % (data_folder, base_name)
out_file_Fs_qty = "%s/%s_Fs_qty.dat" % (data_folder, base_name)

state_file = "%s/state.dat" % (data_folder)

# Get rho, T from state file.
with open(state_file, "r") as f:
    lines = [line.rstrip() for line in f]
    data = lines[1].split(", ")


    model_name = str(data[0])
    T = float(data[1])
    tau_thermostat = float(data[2])
    rho = float(data[3])
    npart = int(data[4])
    dt = float(data[5])


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

    calculate_self_intermediate_scattering_function(traj_file, q_values, out_file=out_file_Fs, out_file_quantities=out_file_Fs_qty, nk=40, dk=0.05)

