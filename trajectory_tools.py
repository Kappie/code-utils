#!/usr/bin/env python
import click
import os
from atooms.trajectory import Trajectory
from my_hoomd_utils import decimate_log_trajectory, detect_base_and_max_exp, detect_and_fix_spacing, decimate_trajectory, get_state_information


@click.group()
def cli():
    pass

@click.command()
@click.option('--traj_file', help='input trajectory file', required=True, type=click.Path())
@click.option('--traj_file_out', help='output trajectory file, relative to directory of input trajectory file.', required=True, type=click.Path())
@click.option('--num_per_block', help='number to keep from each log-spaced block', required=False, type=int)
@click.option('--skip', help='number to skip from linearly spaced traj', required=False, type=int)
def decimate(traj_file, traj_file_out, num_per_block, skip):
    traj_file_out = os.path.join( os.path.dirname(traj_file), traj_file_out )

    with Trajectory(traj_file) as traj:
        result = detect_and_fix_spacing(traj.steps)
        if result['mode'] == "log":
            if num_per_block == None:
                raise Exception("Please supply num_per_block for a log-spaced trajectory.")
            decimated_steps, decimated_frames = decimate_log_trajectory(traj.steps, num_per_block=num_per_block)
        else:
            if skip == None:
                raise Exception("Please supply skip for a linear (or \"other\") trajectory.")
            corrected_steps = result["corrected_steps"]
            decimated_steps, decimated_frames = decimate_trajectory(corrected_steps, skip=skip)

        with Trajectory(traj_file_out, "w") as new_traj:
            for i, frame in enumerate(decimated_frames):
                system = traj[int(frame)]
                new_traj.write(system, decimated_steps[i])

@click.command()
@click.option('--traj_file', help='input trajectory file', required=True, type=click.Path())
def inspect(traj_file):
    state = get_state_information(traj_file)
    dt = state['dt']
    with Trajectory(traj_file) as traj:
        nframes = len(traj.steps)
        species = traj[0].distinct_species()
        nspecies = len(species)
        npart = len(traj[0].particle)
        click.echo("""System information:
    particles: %d
    distinct species: %d""" % (npart, nspecies))
        click.echo("    species: " + ', '.join(species))
        result = detect_and_fix_spacing(traj.steps)
        mode = result["mode"]
        if mode == "log":
            base = result["base"]
            max_exp = result["max_exp"]
            block_size = int(base ** max_exp)
            tmin = traj.steps[0]
            tmax = traj.steps[-1]
            nblocks = (tmax - tmin) // block_size
            click.echo("""Trajectory information:
    frames: %d
    mode: %s
    blocks: %d
    block size: %g = %g * %d = %g * %d ** %d""" % (nframes, mode, nblocks, dt*block_size, dt, block_size, dt, base, max_exp))
        elif mode == "linear":
            click.echo("""Trajectory information:
    frames: %d
    mode: %s
    spacing: %g = %g * %d""" % (nframes, mode, dt*result['spacing'], dt, result["spacing"]))
        elif mode == "other":
            click.echo("""Trajectory information:
    frames: %d
    mode: %s
    steps: %s""" % (nframes, mode, ", ".join( [str(x) for x in result["corrected_steps"].tolist()] )))

            
cli.add_command(decimate)
cli.add_command(inspect)


if __name__ == '__main__':
    cli()
