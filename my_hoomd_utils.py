import math
import numpy as np
import garnett
import MDAnalysis as mda
import io
import gsd
import gsd.hoomd
import subprocess
import re
import atooms.postprocessing as pp
import atooms
from atooms.trajectory.decorators import Sliced
from os.path import basename, dirname, splitext, join, exists



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


def dump_hoomd_snapshot_to_xyz(snap, filename, save_velocity=False, save_force=False, write_mode="w"):
    pos = snap.particles.position
    im = snap.particles.image
    N = pos.shape[0]
    types = snap.particles.typeid
    box = snap.box

    comment_line="columns=type,x,y,z,imx,imy,imz"
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

    comment_line += " cell=%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n" % (box.Lx, box.Ly, box.Lz, box.xy, box.xz, box.yz)
    data = np.column_stack(columns)

    with open(filename, write_mode) as f:
        f.write("%d\n" % N)
        f.write(comment_line)

        np.savetxt(f, data, fmt=fmt)


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


def detect_base_and_max_exp(steps):
    # Always start at 0.
    steps = np.asarray(steps)
    steps -= steps[0]

    base = steps[2] / steps[1]

    n = 1
    while steps[n] == base ** (n-1):
        n += 1

    max_exp = n - 2 

    return base, max_exp

def truncate_trajectory_after_final_whole_block(traj):
    base, max_exp = detect_base_and_max_exp(traj.steps)
    num_frames = len(traj.steps)
    num_blocks = num_frames // (max_exp + 1)
    truncated_traj = Sliced(traj, slice(0, (max_exp+1)*num_blocks + 1))

    return truncated_traj


def calculate_msd(traj_file, num_partitions=1, out_file=None):
    # Preprocess file. Accepted file formats: .gsd, .xyz, .xyz.gz.
    traj_file_orig = traj_file
    if traj_file_orig.endswith(".gsd"):
        traj_file_xyz_gz = traj_file[:-3] + "xyz.gz"
        if exists(traj_file_xyz_gz):
            print(".xyz.gz file of this trajectory already exists")
            traj_file_orig = traj_file_xyz_gz
            traj_file = traj_file_xyz_gz
        else:
            traj_file_xyz = traj_file[:-3] + "xyz"
            print("converting .gsd file to .xyz.")
            convert_trajectory_gsd_to_xyz(traj_file, traj_file_xyz, compress=False)
            traj_file = traj_file_xyz
    if traj_file_orig.endswith(".gz"):
        print("decompressing .gz.")
        subprocess.run(["gzip", "--decompress", "--keep", "--force", traj_file])
        traj_file = splitext(traj_file)[0]
        decompressed_traj_file = traj_file


    # Start analysis.
    ts = []
    msds = []
    tgrid = None

    with atooms.trajectory.Trajectory(traj_file) as traj:
        base, max_exp = detect_base_and_max_exp(traj.steps)
        block_size = int(base ** max_exp)
        print("detected a logarithmically spaced trajectory with block_size %d ** %d" % (base, max_exp))
        nframes = len(traj)
        frames_per_partition = nframes // num_partitions
        print("number of frames to analyze: %d" % nframes)
        species = traj[0].distinct_species()
        print("Detected %d different species, " % len(species), species)
        for n in range(num_partitions):
            print("starting with partition %d/%d" % (n, num_partitions))
            subtraj = Sliced(traj, slice(n*frames_per_partition, (n+1)*frames_per_partition))
            tmin = subtraj.steps[0]
            tmax = subtraj.steps[-1]
            nblocks = (tmax - tmin) // block_size
            # tgrid the same for each tranche. 
            if not tgrid:
                tgrid = [base**m for m in range(0, max_exp)] + [float(m*block_size) for m in range(1, nblocks//2)]
            # analysis = pp.MeanSquareDisplacement(subtraj, tgrid=tgrid)
            analysis = pp.Partial(pp.MeanSquareDisplacement, trajectory=subtraj, tgrid=tgrid, species=species)
            analysis.do()
            analysis = analysis.partial # Dict with results for each species

            ts.append(analysis[species[0]].grid)
            this_msd = []
            for spec in species:
                this_msd.append(analysis[spec].value)
            msds.append(this_msd)

    # Don't keep decompressed file.
    if traj_file_orig.endswith(".gz"):
        print("removing decompressed .xyz file.")
        subprocess.run(["rm", decompressed_traj_file])
    # Compress .xyz file for next time.
    if traj_file_orig.endswith(".gsd") and not exists(traj_file_xyz + ".gz"):
        print("Compressing .xyz file for next time.")
        subprocess.run(["gzip", "--verbose", traj_file_xyz])

    if out_file:
        # Save
        columns = (ts[0],)
        fmt = "%d"
        header = "columns=step,"
        for n in range(num_partitions):
            for spec_i, spec in enumerate(species):
                columns += (msds[n][spec_i],)
                fmt += " %.8g"
                header += "msd_partition%d_species%s," % (n + 1, spec)
        header = header[:-1]    # remove final comma.

        columns = np.column_stack(columns)
        np.savetxt(out_file, columns, fmt=fmt, header=header)

    return ts, msds
        

def calculate_radial_distribution_function(traj_file, out_file=None):
    """Assume trajectory has format .xyz.gz"""

    # decompress.
    subprocess.run(["gzip", "--decompress", "--keep", "--force", traj_file])
    traj_file = splitext(traj_file)[0]

    with atooms.trajectory.Trajectory(traj_file) as traj:
        species = traj[0].distinct_species()

        analysis = pp.Partial(pp.RadialDistributionFunction, trajectory=traj, species=species)
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
    subprocess.run(["rm", traj_file])


def calculate_structure_factor(traj_file, out_file=None, out_file_kmax=None, ksamples=40):
    """
    Assume trajectory has format .xyz.gz
    Writes the first peak of the structure factor for each species in out_file_kmax.
    """

    # decompress.
    subprocess.run(["gzip", "--decompress", "--keep", "--force", traj_file])
    traj_file = splitext(traj_file)[0]

    with atooms.trajectory.Trajectory(traj_file) as traj:
        species = traj[0].distinct_species()

        analysis = pp.Partial(pp.StructureFactor, trajectory=traj, species=species, ksamples=ksamples)
        analysis.do()
        analysis = analysis.partial # Dict with results for each species

        sks = []
        ks = analysis[(species[0], species[0])].grid    # Grid is the same for every combination.
        for spec1 in species:
            for spec2 in species:
                sks.append(analysis[(spec1, spec2)].value)

    if out_file:
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

        columns = np.column_stack(columns)
        np.savetxt(out_file, columns, fmt=fmt, header=header)

    # if out_file_kmax:


    # Don't keep decompressed file.
    subprocess.run(["rm", traj_file])


def calculate_self_intermediate_scattering_function(traj_file, q_values, out_file=None):
    """
    Assume trajectory has format .xyz.gz
    q_values contains a q value for each species; this would normally be the maximum of the first peak in the static structure factor.
    """

    q_values = np.array(q_values)
    # decompress.
    subprocess.run(["gzip", "--decompress", "--keep", "--force", traj_file])
    traj_file = splitext(traj_file)[0]

    with atooms.trajectory.Trajectory(traj_file) as traj:
        # traj = truncate_trajectory_after_final_whole_block(traj)
        num_frames = len(traj.steps)
        base, max_exp = detect_base_and_max_exp(traj.steps)
        traj = Sliced(traj, slice(0, 255*(max_exp) + 1))    # super weird bug in hoomd where it fucks up the period after 256 blocks?
        block_size = int(base ** max_exp)
        print("detected a logarithmically spaced trajectory with block_size %d ** %d = %d" % (base, max_exp, block_size))
        species = traj[0].distinct_species()
        nframes = len(traj)
        tmin = traj.steps[0]
        tmax = traj.steps[-1]
        nblocks = (tmax - tmin) // block_size
        tgrid = [base**m for m in range(0, max_exp)] + [float(2**m*block_size) for m in range(0, int(np.floor(np.log2(nblocks//2))) + 1)]


        # I don't know how to tell the library to compute a different q value for each species.
        # I just compute both q values for each species. Not very efficient.
        analysis = pp.Partial(pp.SelfIntermediateScattering, trajectory=traj, species=species, tgrid=tgrid, kgrid=q_values)
        analysis.do()
        analysis = analysis.partial # Dict with results for each species

        fks = []
        actual_q_values = []
        for i, spec in enumerate(species):
            # This happens if the q values are actually the same for both species.
            if len( analysis[species[0]].kgrid ) == 1:
                fks.append(analysis[spec].value[0])
                actual_q_values.append(analysis[spec].kgrid[0])
            else:
                fks.append(analysis[spec].value[i])
                actual_q_values.append(analysis[spec].kgrid[i])

    if out_file:
        columns = (tgrid,)
        fmt = "%d"
        header = "columns=step,"
        for i, spec in enumerate(species):
            columns += (fks[i],)
            fmt += " %.8g"
            header += "F_s(t, k=%.2f)_species%s," % (actual_q_values[i], spec)
        header = header[:-1]    # remove final comma.

        columns = np.column_stack(columns)
        np.savetxt(out_file, columns, fmt=fmt, header=header)

    # Don't keep decompressed file.
    subprocess.run(["rm", traj_file])







