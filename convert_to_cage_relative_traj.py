#!/usr/bin/env python
import numpy as np
import os
from atooms.trajectory import Trajectory
from lerner_group.trajectory_tools import voronoi_analysis, cage_relative_traj
from lerner_group.read_tools import read_trajectory
from my_hoomd_utils import detect_and_fix_spacing
import argparse


p = argparse.ArgumentParser()
p.add_argument("--traj_file", type=str)
args = p.parse_args()

traj_file = args.traj_file
data_folder = os.path.dirname( traj_file )
base_name = os.path.splitext(os.path.split(traj_file)[1])[0]

max_frames = np.inf
rel_traj_file = "%s/cage_relative_%s.gsd" % (data_folder, base_name)

snap, num_frames = read_trajectory(traj_file, frame=0, unfold=False, fixed_cm=False)
pos_start = snap['pos']
npart = snap['npart']
box_size = snap['box']

num_frames = min([max_frames, num_frames])

print("reading trajectory of %d frames..." % num_frames)
snap, _ = read_trajectory(traj_file, frame=slice(0, num_frames), unfold=False, fixed_cm=False)
pos = snap['pos']
steps = snap['steps']
result = detect_and_fix_spacing(steps)
steps = result["corrected_steps"]
print("done.")



print("calculating relative positions...")
voro = voronoi_analysis(pos_start, box_size)
relative_pos = cage_relative_traj(pos, box_size, voro) 
print("done.")


with Trajectory(traj_file, "r") as traj:
    with Trajectory(rel_traj_file, "w") as rel_traj:

        for frame in range(num_frames):
            system = traj[frame]
            for i in range(npart):
                system.particle[i].position = relative_pos[frame, i, :]

            print("writing %d" % frame)
            rel_traj.write(system, steps[frame])
