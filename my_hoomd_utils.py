import math
import numpy as np
import garnett
import io
import gsd
import gsd.hoomd
import subprocess
import re
import atooms.postprocessing as pp
import os
import atooms
from atooms.trajectory.decorators import Sliced
from atooms.trajectory import Trajectory
from atooms.trajectory.decorators import Unfolded
import atooms.trajectory
from os.path import basename, dirname, splitext, join, exists
from file_reading_functions import make_nested_dir
from scipy.interpolate import interp1d

from atooms.postprocessing.fourierspace import FourierSpaceCorrelation
from atooms.trajectory import Trajectory

try:
    import hoomd
    from hoomd import md
except:
    pass

from collections import namedtuple
from dataclasses import dataclass


HOOMD_TIME_STEP_LIMIT = 1000000000


def log_period(block_size, base=2):
    def period(n):
        max_exp = int(math.log(block_size, base)) + 1
        n_mod = n % max_exp
        n_div = n // max_exp
        if n_mod == 0:
            return n_div * block_size 
        else:
            return n_div * block_size + base**(n_mod - 1)

    return period


def dump_hoomd_snapshot_to_xyz(snap, filename, save_velocity=False, save_force=False, step=0, write_mode="w", compress=True):
    if write_mode[0] == "a":
        compress = False

    pos = snap.particles.position
    im = snap.particles.image
    N = pos.shape[0]
    types = snap.particles.typeid
    box = snap.box

    comment_line="step=%d columns=type,x,y,z,imx,imy,imz" % (step)
    columns = (types, pos[:, 0], pos[:, 1], pos[:, 2], im[:, 0], im[:, 1], im[:, 2])
    fmt = "%d %.12g %.12g %.12g %d %d %d"

    if save_velocity:
        velocity = snap.particles.velocity
        comment_line += ",vx,vy,vz"
        columns += (velocity[:, 0], velocity[:, 1], velocity[:, 2])
        fmt += " %.12g %.12g %.12g" 

    if save_force:
        force = np.multiply( snap.particles.acceleration,  np.matlib.repmat(snap.particles.mass, 3, 1).T )
        comment_line += ",fx,fy,fz"
        columns += (force[:, 0], force[:, 1], force[:, 2])
        fmt += " %.12g %.12g %.12g" 

    # comment_line += " cell=%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n" % (box.Lx, box.Ly, box.Lz, box.xy, box.xz, box.yz)
    comment_line += " cell=%.12g,%.12g,%.12g\n" % (box.Lx, box.Ly, box.Lz)
    data = np.column_stack(columns)

    with open(filename, write_mode) as f:
        f.write("%d\n" % N)
        f.write(comment_line)

        np.savetxt(f, data, fmt=fmt)

    if compress:
        subprocess.run(["gzip", "-f", filename])




def hoomd_snapshot_to_gsd_frame(snap, step=0):
    velocity = snap.particles.velocity
    pos = snap.particles.position
    v = snap.particles.velocity
    im = snap.particles.image
    N = pos.shape[0]
    typeid = snap.particles.typeid
    box = snap.box

    frame = gsd.hoomd.Snapshot()
    frame.configuration.step = step
    frame.particles.types = snap.particles.types
    frame.particles.N = N
    frame.particles.position = pos
    frame.particles.velocity = v
    frame.particles.typeid = typeid
    frame.particles.image = im
    frame.particles.mass = snap.particles.mass
    frame.configuration.box = [box.Lx, box.Ly, box.Lz, box.xy, box.xz, box.yz]

    return frame

def xyz_to_edan_format(in_filename, out_filename):
    # Edan's format:
    # L, strain, dummy, dummy
    # columns=x,y,z,type

    with open(in_filename) as f:
        first_line = f.readline()   # useless, just contains N
        comment_line = f.readline()
    # Get L.
    p = re.compile("cell=([\.\d]+)")
    result = p.search(comment_line)
    L = float(result.group(1))

    data = np.loadtxt(in_filename, skiprows=2)
    types = data[:, 0].astype(int)
    radii = np.array( [ 0.5 if types[i] == 0 else 0.7 for i in range(types.size) ] )
    pos = (data[:, 1:4] + L/2) / L

    columns = np.column_stack((pos[:, 0], pos[:, 1], pos[:, 2], radii))
    fmt = "%.12g %.12g %.12g %.12g"
    header = "%.12g 0 0 0" % L
    np.savetxt(out_filename, columns, header=header, fmt=fmt, comments='')


def edan_format_to_gsd_hertzian(in_filename, out_filename):
    with open(in_filename) as f:
        first_line = f.readline()   
    L = float( first_line.split("\t")[0] )

    data = np.loadtxt(in_filename, skiprows=1)
    N = data.shape[0]
    pos = data[:, 0:3]
    v = data[:, 3:6]
    radii = data[:, 6]
    typeid = np.array( [0 if radii[i] == 0.5 else 1 for i in range(radii.size)], dtype=int) # This is very Hertzian specific.

    snap = gsd.hoomd.Snapshot()
    snap.configuration.step = 0
    snap.particles.types = ['A', 'B']
    snap.particles.N = N
    snap.particles.position = L*pos - L/2
    snap.particles.velocity = v
    snap.particles.typeid = typeid
    snap.configuration.box = [L, L, L, 0, 0, 0] # Assume all strains 0.

    with gsd.hoomd.open(name=out_filename, mode='wb') as traj:
        traj.append(snap)


def edan_format_to_gsd_ipl(in_filename, out_filename):
    with open(in_filename) as f:
        first_line = f.readline()   
    L = float( first_line.split("\t")[0] )

    data = np.loadtxt(in_filename, skiprows=1)
    N = data.shape[0]
    pos = data[:, 0:3]
    v = data[:, 3:6]
    typeid = data[:, 6].astype(int)

    snap = gsd.hoomd.Snapshot()
    snap.configuration.step = 0
    snap.particles.types = ['A', 'B']
    snap.particles.N = N
    snap.particles.position = L*pos - L/2
    snap.particles.velocity = v
    snap.particles.typeid = typeid
    snap.configuration.box = [L, L, L, 0, 0, 0] # Assume all strains 0.

    with gsd.hoomd.open(name=out_filename, mode='wb') as traj:
        traj.append(snap)


def convert_frame_gsd_to_xyz(frame, filename_or_handle):
    """GSD snapshots are somehow slightly different from Hoomd-blue snapshots, so I have to essentially rewrite the above function. 
       There are no forces, and this is necessarily float32, because the gsd frame is always single precision."""

    box = frame.configuration.box
    im = frame.particles.image
    pos = frame.particles.position
    # Unwrap positions.
    for d in range(3):
        pos[:, d] += im[:, d] * box[d]
    v = frame.particles.velocity
    types = np.array(frame.particles.typeid, dtype=np.object)  # Have to do this for column stack to be able to stack arrays with different data types.
    time_step = frame.configuration.step
    N = pos.shape[0]

    comment_line = "step=%d columns=type,x,y,z,vx,vy,vz cell=%.12g,%.12g,%.12g\n" % (time_step, box[0], box[1], box[2])
    columns = (types, pos[:, 0], pos[:, 1], pos[:, 2], v[:, 0], v[:, 1], v[:, 2])
    fmt = "%s %.7g %.7g %.7g %.7g %.7g %.7g"
    data = np.column_stack(columns)

    # Create file if I receive a string, otherwise assume I already have a filehandle.
    if isinstance(filename_or_handle, str): 
        f = open(filename_or_handle, "w")
    else:
        f = filename_or_handle

    f.write("%d\n" % N)
    f.write(comment_line)
    np.savetxt(f, data, fmt=fmt)

    if isinstance(filename_or_handle, str): 
        f.close()



def convert_trajectory_gsd_to_xyz(infile, outfile, compress=True):
    with gsd.hoomd.open(infile, mode='rb') as traj:
        with open(outfile, "w") as f:
            for frame in traj:
                convert_frame_gsd_to_xyz(frame, f)
    
    if compress:
        subprocess.run(["gzip", "--verbose", outfile])


def read_trajectory_xyz(traj_file):
    if traj_file.endswith(".gz"):
        subprocess.run(["gzip", "--decompress", "--keep", "--force", traj_file])
        traj_file = splitext(trajectory_file)[0]   # chop .gz
        compressed = True
    else:
        compressed = False

    with open(traj_file, "r") as f:
        N = int( f.readline().rstrip('\n') )
        comment_line = f.readline().rstrip('\n')
        p = re.compile("cell=(.*)")
        result = p.search(comment_line)
        box = result.group(1).split(",")
        Lx, Ly, Lz = np.float64(box[0]), np.float64(box[1]), np.float64(box[2])

        nlines = len(f.readlines())
        nframes = nlines // (N + 2)     # Each snapshot has the system size as a first line, and a comment line after that.
        ptypes = np.zeros((nframes, N), dtype=int)
        pos = np.zeros((nframes, N, 3))

        f.seek(0)   # Go back to start of file.
        for frame in range(nframes):
            # Don't use first two lines.
            f.readline(); f.readline()
            for i in range(N):
                data = f.readline().rstrip('\n').split()
                ptypes[frame, i], pos[frame, i, 0], pos[frame, i, 1], pos[frame, i, 2] = int(data[0]), np.float64(data[1]), np.float64(data[2]), np.float64(data[3])

    # Shift positions.
    pos[:, :, 0] += Lx/2
    pos[:, :, 1] += Ly/2
    pos[:, :, 2] += Lz/2

    if Lz == 1.0:   # 2D
        pos = pos[:, :, :2]
        box = np.array([Lx, Ly])
    else:
        box = np.array([Lx, Ly, Lz])

    return ( N, pos, ptypes, box )



def read_trajectory(traj_file, frame=0, unfold=False):
    # Unzip if it is a compressed format
    file_extension = splitext(traj_file)[1]
    if file_extension == ".gz":
        subprocess.run(["gzip", "--decompress", "--keep", "--force", traj_file])
        traj_file = splitext(traj_file)[0]

    with atooms.trajectory.Trajectory(traj_file) as traj:
        num_frames = len(traj)
        if unfold:
            traj_unf = Unfolded(traj)
            system_unf = traj_unf[frame]
            pos_unf = system_unf.dump(['pos'])

        system = traj[frame]
        step = traj.steps[frame]

        data  = system.dump(['pos', 'spe'])
        box = system.cell.side
        if traj[0].number_of_dimensions == 2 or box[2] == 1: # HOOMD convention for 2D is to have a [Lx, Ly, 1] box.
            ndim = 2
        else:
            ndim = 3
        box = box[:ndim]

        N = len(system.particle)
        distinct_species = system.distinct_species()

        # Convert species from ['A', 'A', 'B', ...] to [0, 0, 1, ...] when distinct_species = ['A', 'B']
        species_to_typeid = {}
        typeid = 0
        for species in distinct_species:
            species_to_typeid[species] = typeid
            typeid += 1

        # Convert positions from -L/2, L/2 to 0, L
        pos = data['particle.position'][:, :ndim]
        for d in range(pos.shape[1]):
            pos[:, d] += box[d]/2
            if unfold:
                pos_unf[:, d] += box[d]/2

        species = data['particle.species']    # This is 'A', 'A', 'B', etc when distinct_species = ['A', 'B']
        convert_to_typeid = np.vectorize( lambda species: species_to_typeid[species] )
        typeid = convert_to_typeid(species).astype(int) # This is 0, 0, 1, etc when distinct_species = ['A', 'B']

    snap = {'step': step, 'npart': N, 'pos': pos, 'ptypes': typeid, 'box': box}
    if unfold:
        snap['pos_unf'] = pos_unf

    # Remove unzipped file.
    if file_extension == ".gz":
        os.remove(traj_file)

    return snap, num_frames


def detect_base_and_max_exp(steps):
    # Always start at 0.
    steps = np.asarray(steps)
    steps -= steps[0]

    base = steps[2] / steps[1]

    n = 1
    while steps[n] == base ** (n-1):
        n += 1

    max_exp = n - 2 
    block_length = max_exp + 1
    block_size = int(base ** max_exp)

    # Now, correct the time steps of subsequent simulation chunks.
    start_chunks = np.nonzero(steps == 0)[0][1:]    # Throw away first start, which is always at index 0.
    corrected_steps = np.copy(steps)
    if start_chunks.size != 0:
        chunk_length = start_chunks[0]
        chunk_size = (chunk_length // block_length) * block_size  # This assumes a simulation chunk is always an integer multiple of block_size.
        corrected_steps = np.copy(steps)
        for start_index in start_chunks:
            corrected_steps[start_index:] += chunk_size

    return base, max_exp, corrected_steps

def detect_and_fix_spacing(steps):
    """
    This can detect linear and logarithmic spacing, and corrects the resetting of steps that occurs if a simulation consists of multiple chunks.
    """
    # Always start at 0.
    steps = np.asarray(steps)
    steps -= steps[0]

    diff1 = steps[2] - steps[1]
    diff2 = steps[3] - steps[2]

    result = {}

    if diff1 == diff2:
        result["mode"] = "linear"
        result["spacing"] = diff1
    else:
        result["mode"] = "log"

        base = steps[2] / steps[1]

        n = 1
        while steps[n] == base ** (n-1):
            n += 1

        max_exp = n - 2 
        block_length = max_exp + 1
        block_size = int(base ** max_exp)

        result["base"] = base
        result["max_exp"] = max_exp

    # Now, correct the time steps of subsequent simulation chunks.
    start_chunks = np.nonzero(steps == 0)[0][1:]    # Throw away first start, which is always at index 0.
    corrected_steps = np.copy(steps)
    if start_chunks.size != 0:
        chunk_length = start_chunks[0]
        if result["mode"] == "log":
            chunk_size = (chunk_length // block_length) * block_size  # This assumes a simulation chunk is always an integer multiple of block_size.
        else:
            chunk_size = chunk_length * result["spacing"]

        for start_index in start_chunks:
            corrected_steps[start_index:] += chunk_size

    result["corrected_steps"] = corrected_steps

    return result


def truncate_trajectory_after_final_whole_block(traj):
    base, max_exp = detect_base_and_max_exp(traj.steps)
    num_frames = len(traj.steps)
    num_blocks = num_frames // (max_exp + 1)
    truncated_traj = Sliced(traj, slice(0, (max_exp+1)*num_blocks + 1))

    return truncated_traj


def calculate_msd(traj_file, num_partitions=1, out_file=None, out_file_quantities=None):
    ts = []
    msds = []
    derived_quantities = [] # contains dicts with diffusive time and diffusion coefficient.
    tgrid = None

    with atooms.trajectory.Trajectory(traj_file) as traj:
        nframes = len(traj)
        frames_per_partition = nframes // num_partitions
        print("number of frames to analyze: %d" % nframes)
        species = traj[0].distinct_species()
        print("Detected %d different species, " % len(species), species)
        # base, max_exp, corrected_steps = detect_base_and_max_exp(traj.steps)
        result = detect_and_fix_spacing(traj.steps)
        corrected_steps = result["corrected_steps"]
        traj.steps = corrected_steps.tolist()
        mode = result["mode"]
        if mode == "log":
            base = result["base"]
            max_exp = result["max_exp"]
            block_size = int(base ** max_exp)
            print("detected a logarithmically spaced trajectory with block_size %d ** %d. Num frames in block: %d" % (base, max_exp, traj.block_size))
        elif mode == "linear":
            spacing = result["spacing"]
            print("Detected a linearly spaced trajectory with spacing %d" % (spacing))

        for n in range(num_partitions):
            print("starting with partition %d/%d" % (n+1, num_partitions))
            subtraj = Sliced(traj, slice(n*frames_per_partition, (n+1)*frames_per_partition))
            if mode == "log":
                tmin = subtraj.steps[0]
                tmax = subtraj.steps[-1]
                nblocks = (tmax - tmin) // block_size
                # tgrid the same for each tranche. 
                if not tgrid:
                    tgrid = [base**m for m in range(0, max_exp)] + [float(m*block_size) for m in range(1, int(nblocks//2))]
            elif mode == "linear":
                if not tgrid:
                    tgrid = subtraj.steps

            analysis = pp.Partial(pp.MeanSquareDisplacement, trajectory=subtraj, tgrid=tgrid, species=species)
            analysis.do()
            analysis = analysis.partial # Dict with results for each species

            ts.append(analysis[species[0]].grid)
            this_msd = []
            this_derived_quantities = []
            for spec in species:
                this_msd.append(analysis[spec].value)
                this_derived_quantities.append( analysis[spec].analysis )
            msds.append(this_msd)
            derived_quantities.append(this_derived_quantities)


    if out_file:
        # Save
        columns = (ts[0],)
        fmt = "%d"
        header = "columns=step,"
        tau_D = np.zeros((num_partitions, len(species)))
        D = np.zeros((num_partitions, len(species)))
        min_signal_len = np.inf
        for n in range(num_partitions):
            for spec_i, spec in enumerate(species):
                columns += (msds[n][spec_i],)
                signal_length = len(columns[-1])
                if signal_length < min_signal_len:
                    min_signal_len = signal_length
                fmt += " %.8g"
                header += "msd_partition%d_species%s," % (n + 1, spec)
                tau_D[n, spec_i] = derived_quantities[n][spec_i].get('diffusive time tau_D', 0.0)
                D[n, spec_i] = derived_quantities[n][spec_i].get('diffusion coefficient D', 0.0)

        new_columns = ()
        for i in range( len(columns) ):
            new_columns += (columns[i][:min_signal_len],)

        header = header[:-1]    # remove final comma.
        columns = np.column_stack(new_columns)
        np.savetxt(out_file, columns, fmt=fmt, header=header)

    if out_file_quantities:
        header = "columns="
        fmt = ""
        columns = ()
        for spec_i, spec in enumerate(species):
            header += "D_%s,tau_D_%s," % (spec, spec)
            fmt += "%.6g %.6g "
            columns += (D[:, spec_i],tau_D[:, spec_i])

        header = header[:-1]    # remove final comma.
        columns = np.column_stack(columns)
        np.savetxt(out_file_quantities, columns, fmt=fmt.strip(), header=header)

    return ts, msds
        

def calculate_radial_distribution_function(traj_file, out_file=None):

    with atooms.trajectory.Trajectory(traj_file) as traj:
        print("Averaging over %d snapshots" % len(traj))
        species = traj[0].distinct_species()

        analysis = pp.Partial(pp.RadialDistributionFunction, trajectory=traj, species=species, dr=0.02)
        analysis.do()
        analysis = analysis.partial # Dict with results for each species

        grs = []
        rs = analysis[(species[0], species[0])].grid    # Grid is the same for every combination.
        for spec1 in species:
            for spec2 in species:
                grs.append(analysis[(spec1, spec2)].value)

    if out_file:
        columns = (rs,)
        fmt = "%.5g"
        header = "columns=r,"
        counter = 0
        for i, spec1 in enumerate(species):
            for j, spec2 in enumerate(species):
                columns += (grs[counter],)
                fmt += " %.8g"
                header += "gr_%s-%s," % (spec1, spec2)
                counter += 1
        header = header[:-1]    # remove final comma.

        columns = np.column_stack(columns)
        np.savetxt(out_file, columns, fmt=fmt, header=header)

    # Don't keep decompressed file.
    # subprocess.run(["rm", traj_file])


def calculate_structure_factor(traj_file, out_file=None, mode="separate_species", **kwargs):
    """
    Assume trajectory has format .xyz.gz
    Writes the first peak of the structure factor for each species in out_file_kmax.
    """

    # decompress.
    # subprocess.run(["gzip", "--decompress", "--keep", "--force", traj_file])
    # traj_file = splitext(traj_file)[0]

    with atooms.trajectory.Trajectory(traj_file) as traj:
        if mode == "separate_species":

            species = traj[0].distinct_species()
            analysis = pp.Partial(pp.StructureFactor, trajectory=traj, species=species, **kwargs)
            analysis.do()
            analysis = analysis.partial # Dict with results for each species

            sks = []
            ks = analysis[(species[0], species[0])].grid    # Grid is the same for every combination.
            for spec1 in species:
                for spec2 in species:
                    sks.append(analysis[(spec1, spec2)].value)
        elif mode == "mix_species":
            analysis = pp.StructureFactorLegacy(traj, **kwargs)
            analysis.do()


    if out_file:
        if mode == "separate_species":
            columns = (ks,)
            fmt = "%.5g"
            header = "columns=k,"
            counter = 0
            for i, spec1 in enumerate(species):
                for j, spec2 in enumerate(species):
                    columns += (sks[counter],)
                    fmt += " %.8g"
                    header += "Sk_%s-%s," % (spec1, spec2)
                    counter += 1
            header = header[:-1]    # remove final comma.
        elif mode == "mix_species":
            columns = (analysis.grid, analysis.value)
            fmt = "%.5g %.5g"
            header = "columns=k,S"

        columns = np.column_stack(columns)
        np.savetxt(out_file, columns, fmt=fmt, header=header)

def _extract_tau(fks, t):
    cutoff = 0.2
    boundaries_fit = np.asarray([0.05, 0.7])
    num_species = len(fks)
    taus = np.zeros(num_species)

    for i, fk in enumerate(fks):
        # import pdb; pdb.set_trace()
        idx = np.logical_and(fk > boundaries_fit[0], fk < boundaries_fit[1])
        fk_zoom = fk[idx]
        t_zoom = t[idx]

        try:
            interp_function = interp1d(fk_zoom, np.log(t_zoom))
            taus[i] = np.exp(interp_function(cutoff))
        except ValueError:
            taus[i] = np.nan

    return taus

def calculate_self_intermediate_scattering_function(traj_file, k_values, out_file=None, out_file_quantities=None, **kwargs):
    """
    Assume trajectory has format .xyz.gz
    k_values contains a q value for each species; this would normally be the maximum of the first peak in the static structure factor.
    """

    # This is necessary because atooms-pp has bugs when k_values is not sorted in ascending order.
    # But we need to remember which k-value belongs to which species.
    k_values = np.array(k_values)
    unsorted_k_values = np.copy(k_values)
    sort_idx = np.argsort(k_values)
    k_values = k_values[sort_idx]

    with atooms.trajectory.Trajectory(traj_file) as traj:
        nframes = len(traj.steps)
        species = traj[0].distinct_species()
        print("Using q_values", unsorted_k_values, "for species", species)

        result = detect_and_fix_spacing(traj.steps)
        corrected_steps = result["corrected_steps"]
        traj.steps = corrected_steps.tolist()
        mode = result["mode"]
        if mode == "log":
            base = result["base"]
            max_exp = result["max_exp"]
            block_size = int(base ** max_exp)
            print("detected a logarithmically spaced trajectory with block_size %d ** %d. Num frames in block: %d" % (base, max_exp, traj.block_size))
        elif mode == "linear":
            spacing = result["spacing"]

        if mode == "log":
            tmin = traj.steps[0]
            tmax = traj.steps[-1]
            nblocks = (tmax - tmin) // block_size
            # tgrid the same for each tranche. 
            tgrid = [base**m for m in range(0, max_exp)] + [float(m*block_size) for m in range(1, int(nblocks//2))]
        elif mode == "linear":
            tgrid = traj.steps

        # I don't know how to tell the library to compute a different q value for each species.
        # I just compute both q values for each species. Not very efficient.
        analysis = pp.Partial(pp.SelfIntermediateScattering, trajectory=traj, species=species, tgrid=tgrid, kgrid=k_values, **kwargs)
        analysis.do()
        analysis = analysis.partial # Dict with results for each species

        fks = []
        actual_k_values = []
        for i, spec in enumerate(species):
            # This happens if the q values are actually the same for all species.
            if len( analysis[species[0]].kgrid ) == 1:
                fks.append( np.array(analysis[spec].value[0]) )
                actual_k_values.append(analysis[spec].kgrid[0])
            else:
                fks.append( np.array(analysis[spec].value[sort_idx[i]]) )
                actual_k_values.append(analysis[spec].kgrid[sort_idx[i]])


    print("Actual kgrid:", actual_k_values)
    tgrid = np.array(tgrid, dtype=int)
    taus = _extract_tau(fks, tgrid)
    print("tau:", taus)

    if out_file:
        columns = (tgrid,)
        fmt = "%d"
        header = "columns=step,"
        for i, spec in enumerate(species):
            columns += (fks[i],)
            fmt += " %.8g"
            header += "F_s(t, k=%.2f)_species%s," % (actual_k_values[i], spec)
        header = header[:-1]    # remove final comma.

        columns = np.column_stack(columns)
        np.savetxt(out_file, columns, fmt=fmt, header=header)


    if out_file_quantities:
        header = "# columns="
        data = ""
        for spec_i, spec in enumerate(species):
            header += "tau_%s,q_%s," % (spec, spec)
            data += "%.6g %.3g " % (taus[spec_i], actual_k_values[spec_i])
        header += "\n"
        data += "\n"

        with open(out_file_quantities, "w") as f:
            f.write(header)
            f.write(data)


def self_intermediate_scattering_function_particle_iso_avg(traj_file, k, start_pos, final_pos, **kwargs):
    """
    traj_file: is required to get box length, etc. (was not really necessary, but is convenient for now.)
    final_pos: (num_iso_runs, N, dim) array
    """

    tgrid = [0, 1]
    kgrid = [k]
    npart = start_pos.shape[0]
    ndim = start_pos.shape[1]
    num_iso_runs = final_pos.shape[0]

    with Trajectory(traj_file, "r") as traj:
        corr = FourierSpaceCorrelation(trajectory=traj, grid=[kgrid, tgrid], **kwargs)

    corr.kgrid = kgrid

    # Setup grid once. If cell changes we'll call it again
    corr._setup()
    # Pick up a random, unique set of nk vectors out of the available ones
    # without exceeding maximum number of vectors in shell nkmax
    corr.kgrid, corr.selection = corr._decimate_k()
    # We redefine the grid because of slight differences on the
    # average k norms appear after decimation.
    corr.kgrid = corr._actual_k_grid()
    # We must fix the keys: just pop them to the their new positions
    # We sort both of them (better check len's)
    for k, kv in zip(sorted(corr.kgrid), sorted(corr.kvector)):
        corr.kvector[k] = corr.kvector.pop(kv)

    kvectors = corr.kvectors[corr.kgrid[0]]
    nk_actual = len(kvectors)


    result = np.zeros((nk_actual, num_iso_runs, npart))
    for k_idx, k in enumerate(kvectors):
        result[k_idx, :, :] = np.cos( np.dot(final_pos - start_pos, k) )

    # Average over different k vectors.
    result = np.mean(result, axis=0)

    return result


# def calculate_correlator_1d(ts, values):
#     base, max_exp = detect_base_and_max_exp(ts)
#     block_size = int(base ** max_exp)
#     tmin = ts[0]
#     tmax = ts[-1]
#     nblocks = (tmax - tmin) // block_size
#     num_frames_in_block = int(max_exp + 1)
#     tgrid = [base**m for m in range(0, max_exp)] + [float(2**m*block_size) for m in range(0, int(np.floor(np.log2(nblocks//2))) + 1)]

#     MockTrajectory = namedtuple('MockTrajectory', 'steps block_size timestep')
#     traj = MockTrajectory(ts, num_frames_in_block, 1)
#     offsets = pp.helpers.setup_t_grid(traj, tgrid)    
#     skip = pp.helpers.adjust_skip(traj, n_origins=-1)

#     def autocorrelator(x, y):
#         return x*y

#     grid, corr = pp.correlation.gcf_offset(autocorrelator, offsets, skip, ts, values)
#     return grid, corr



def setup_cubic_grid_random_types(N, rho, dim=3, particle_types=['A', 'B']):
    L = (N / rho) ** (1/dim)
    n = int(np.ceil(N ** (1/dim)))
    a = L / n

    snapshot = hoomd.data.make_snapshot(N=N, box=hoomd.data.boxdim(L=L, dimensions=dim), particle_types=particle_types, dtype='double')

    # Select half to make type B.
    types = np.zeros((N)).astype(int)
    num_types = len(particle_types)
    num_per_type = N // num_types
    for i in range(N):
        types[i] = i // num_per_type
    np.random.shuffle(types)
    for i in range(N):
        snapshot.particles.typeid[i] = types[i]

    i = 0
    if dim == 3:
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    if i >= N:
                        break

                    snapshot.particles.position[i, 0] = a*x - L/2;
                    snapshot.particles.position[i, 1] = a*y - L/2;
                    snapshot.particles.position[i, 2] = a*z - L/2;
                    i += 1

    elif dim == 2:
        for x in range(n):
            for y in range(n):
                if i >= N:
                    break

                snapshot.particles.position[i, 0] = a*x - L/2;
                snapshot.particles.position[i, 1] = a*y - L/2;
                i += 1


    return snapshot


def setup_output_folders(base_folder):

    traj_folder = os.path.join(base_folder, "trajectory")
    log_folder = os.path.join(base_folder, "log")
    make_nested_dir(traj_folder)
    make_nested_dir(log_folder)
    traj_file = os.path.join(traj_folder, "trajectory.gsd")
    qty_file = os.path.join(log_folder, "quantities.dat")
    state_file = os.path.join(log_folder, "state.dat")
    final_state_file = os.path.join(traj_folder, "final_state.gsd")
    restart_file = os.path.join(traj_folder, r"restart%d.gsd")


    return traj_file, qty_file, final_state_file, state_file, restart_file


