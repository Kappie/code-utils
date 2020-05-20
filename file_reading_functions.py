import numpy as np
import os
import sys


base_data_dir = os.environ.get("MD_DATA_DIR")

def read_modes(dim, N, delta, epsilon, num_of_modes, serialStart, serialEnd):
    datafolder = "%s/ipl/modes/%dD/%d" % (base_data_dir, dim, N)
    if epsilon == 0:
        datafile = "analysis_ipl_modes%dD_N%d_delta%g.dat" % (dim, N, delta)
    else:
        datafile = "analysis_ipl_modes%dD_N%d_delta%g_epsilon%g.dat" % (dim, N, delta, epsilon)

    data = np.loadtxt(os.path.join(datafolder, datafile))

    omega = np.sqrt( np.abs(data[:, 0]) ) 
    omega = np.reshape(omega, ( -1, num_of_modes ))
    participation = np.reshape(data[:, 1], (-1, num_of_modes))
    omega = omega[:, dim:]
    participation = participation[:, dim:]
    return omega[serialStart:serialEnd, :], participation[serialStart:serialEnd, :]


def read_quantities(dim, N, delta, epsilon, serialStart, serialEnd):
    datafolder = "%s/ipl/slowlyQuenchedSolids/%dD" % (base_data_dir, dim)
    if epsilon == 0:
        qty_file = "quantities%dD_N%d_delta%g_%d-%d.dat" % (dim, N, delta, serialStart, serialEnd)
    else:
        qty_file = "quantities%dD_N%d_delta%g_epsilon%g_%d-%d.dat" % (dim, N, delta, epsilon, serialStart, serialEnd)

    quantities = np.loadtxt(os.path.join(datafolder, qty_file))
    G = np.mean(quantities[:, 1])
    K = np.mean(quantities[:, 2])
    return quantities, G, K


def read_eigenvectors(coordinates_file, vectors_file):
    coordinates = np.genfromtxt(coordinates_file, skip_header=1)[:, 0:2]
    vectors = np.genfromtxt(vectors_file, delimiter=" ", dtype=float)
    X = coordinates[:, 0]
    Y = coordinates[:, 1]

    # Odd elements are X components of each particle.
    U = vectors[:, ::2]
    # Even elements are Y components of each particle.
    V = vectors[:, 1::2]

    return ((X, Y), (U, V))


def make_nested_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)




   
