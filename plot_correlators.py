#!/usr/bin/env python

from plotting_functions import *
from datetime import datetime
import argparse
import re
import numpy as np

import atooms
import atooms.trajectory
import gsd.hoomd
import model_utils


p = argparse.ArgumentParser()
p.add_argument("--traj_file", type=str)
p.add_argument("--wu_correlations", action="store_true")
p.add_argument("--T", type=float)
p.add_argument("--rho", type=float)
p.add_argument("--model_name", type=str)
p.add_argument("--npart", type=int)
args = p.parse_args()

traj_file = args.traj_file
wu_correlations = args.wu_correlations

data_folder = os.path.join( os.path.split(os.path.dirname(traj_file))[0], "log")
base_name = os.path.splitext(os.path.basename(traj_file))[0]
data_file_msd = "%s/%s_msd.dat" % (data_folder, base_name)
qty_file_msd = "%s/%s_msd_qty.dat" % (data_folder, base_name)
data_file_gr = "%s/%s_gr.dat" % (data_folder, base_name)
data_file_Sk = "%s/%s_Sk.dat" % (data_folder, base_name)
data_file_Fs = "%s/%s_Fs.dat" % (data_folder, base_name)
data_file_Q = "%s/%s_Q.dat" % (data_folder, base_name)
data_file_chi4_Qs = "%s/%s_chi4_Qs.dat" % (data_folder, base_name)
data_file_chi4_Fs = "%s/%s_chi4_Fs.dat" % (data_folder, base_name)
qty_file_Fs = "%s/%s_Fs_qty.dat" % (data_folder, base_name)
qty_file_Q = "%s/%s_Q_qty.dat" % (data_folder, base_name)
data_file_qty = "%s/quantities.dat" % (data_folder)
state_file = "%s/state.dat" % (data_folder)


msd = os.path.isfile(data_file_msd)
gr = os.path.isfile(data_file_gr)
Sk = os.path.isfile(data_file_Sk)
Fs = os.path.isfile(data_file_Fs)
Q  = os.path.isfile(data_file_Q)
chi4_Qs  = os.path.isfile(data_file_chi4_Qs)
chi4_Fs  = os.path.isfile(data_file_chi4_Fs)
quantities = os.path.isfile(data_file_qty)

num_axes = np.count_nonzero([msd, gr, Sk, Fs, Q, quantities, chi4_Qs, chi4_Fs])
axes = init_fig(grid=(1,num_axes))
current_axis = 0

# Get rho, T, dt from state file.
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
    model = model_utils.models[model_name](T=T, rho=rho)
else:
    T = args.T
    rho = args.rho
    npart = args.npart
    model_name = args.model_name
    model = model_utils.models[model_name](T=T, rho=rho)
    dt = model.get_dt()



if quantities and wu_correlations:  # assume it is always there
    ax = axes[current_axis]
    current_axis += 1

    data = np.loadtxt(data_file_qty, skiprows=1)
    time_last_modified_traj = os.path.getmtime(data_file_qty)
    time_last_modified_msd  = os.path.getmtime(data_file_msd)
    time_last_modified_traj_str = datetime.utcfromtimestamp(time_last_modified_traj).strftime('%m-%d %H:%M')
    time_last_modified_msd_str = datetime.utcfromtimestamp(time_last_modified_msd).strftime('%m-%d %H:%M')
    step = data[:, 0]
    U    = data[:, 1]
    Ekin = data[:, 2]
    T_inst    = data[:, 3]
    p    = data[:, 4]

    V = npart / rho
    W = p*V - npart*T_inst
    U /= npart
    W /= npart
    t = step * dt

    mean_U = np.mean(U)
    mean_W = np.mean(W)
    std_U = np.std(U)
    std_W = np.std(W)
    delta_U = U - mean_U
    delta_W = W - mean_W

    R = np.mean( delta_W * delta_U ) / ( std_U * std_W )
    gamma, intersect = np.polyfit(U, W, 1)

    # nice_plot(ax, U, W)

    ax.set(xlabel="$U/N$", ylabel="$W/N$")

    ax.text(0.5, 0.2, "%s\n$(\\rho, T) = (%.2f, %.2f)$\n$R = %.4f$, $\gamma = %.2f$,\nintersect = %.2f\nmsd date: %s\ntraj date: %s" % (model_name.replace("_", " "), rho, T, R, gamma, intersect, time_last_modified_msd_str, time_last_modified_traj_str), transform=ax.transAxes)
    xlim2 = np.array(ax.get_xlim())
    ax.plot(xlim2, gamma*xlim2 + intersect, ls='--', color='k', lw=fit_lw)


# MSD
if msd:
    ax = axes[current_axis]
    current_axis += 1

    # columns=step,msd_partition1_species0,msd_partition1_species1
    data = np.loadtxt(data_file_msd)
    # columns=D_A,tau_D_A,D_B,tau_D_B
    data_qty = np.loadtxt(qty_file_msd)
    if data_qty.ndim == 1:
        data_qty = np.reshape( data_qty, (1, -1) )
    num_partitions = int((data.shape[1] - 1) / 2)
    step = data[:, 0]
    t = step * dt
    for i in range(1, num_partitions+1):
        # import pdb; pdb.set_trace()
        msd0 = data[:, 2*i - 1]
        msd1 = data[:, 2*i]
        D0 = data_qty[i-1, 0] / dt
        tau_D0 = data_qty[i-1, 1] * dt # tau is given in steps
        D1 = data_qty[i-1, 2] / dt
        tau_D1 = data_qty[i-1, 3] * dt
        nice_plot(ax, t, msd0, color=colors[2*i-2], label=r"A%d" % (i), ls='-')
        nice_plot(ax, t, msd1, color=colors[2*i-1], label=r"B%d" % (i), ls='-')


    ax.set(xlabel="$t$", ylabel="MSD", xscale='log', yscale='log')
    nice_legend(ax)


# g(r)
if gr:
    ax = axes[current_axis]
    current_axis += 1

    # columns=r,gr_0-0,gr_0-1,gr_1-0,gr_1-1
    data = np.loadtxt(data_file_gr)
    r = data[:, 0]
    gr00 = data[:, 1]
    gr01 = data[:, 2]
    gr11 = data[:, 4]

    nice_plot(ax, r, gr00, color=colors[0], label="AA", ls='-')
    nice_plot(ax, r, gr01, color=colors[1], label="AB", ls='-')
    nice_plot(ax, r, gr11, color=colors[2], label="BB", ls='-')
    ax.set(xlabel="$r$", ylabel="Pair distribution function", xlim=(0, 4))
    nice_legend(ax)

# Structure factor
if Sk:
    ax = axes[current_axis]
    current_axis += 1
    # columns=k,Sk_0-0,Sk_0-1,Sk_1-0,Sk_1-1
    data = np.loadtxt(data_file_Sk)
    k = data[:, 0]
    Sk00 = data[:, 1]
    Sk01 = data[:, 2]
    Sk11 = data[:, 4]

    nice_plot(ax, k, Sk00, color=colors[0], label="AA", ls='-')
    nice_plot(ax, k, Sk01, color=colors[1], label="AB", ls='-')
    nice_plot(ax, k, Sk11, color=colors[2], label="BB", ls='-')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    top = np.max(Sk11[k > 0.4])
    ax.set(xlabel="$k$", ylabel="Static Structure Factor", xlim=(0.4, xlim[1]), ylim=(ylim[0], 1.6*top))
    nice_legend(ax)

# Self-part of intermediate scattering function.
if Fs:
    ax = axes[current_axis]
    current_axis += 1

    # Fs
    # columns=step,F_s(t, k=kmax00)_species0,F_s(t, k=kmax11)_species1
    data = np.loadtxt(data_file_Fs)
    step = data[:, 0]
    Fs0 = data[:, 1]
    Fs1 = data[:, 2]
    t = step * dt

    # get kmax from header.
    with open(data_file_Fs) as f:
        header = f.readline()
        result = re.findall(r'k=(.*?)\)', header)
        kmaxAA, kmaxBB = float(result[0]), float(result[1])

    with open(qty_file_Fs) as f:
        header = f.readline()
        data = f.readline().split(" ")
        tauA = float(data[0]) * dt  # tau is given in steps
        tauB = float(data[2]) * dt

    nice_plot(ax, t, Fs0, color=colors[0], label=r"A ($\tau$ = %.3g)" % tauA, ls='-')
    nice_plot(ax, t, Fs1, color=colors[2], label=r"B ($\tau$ = %.3g)" % tauB, ls='-')

    ax.axvline(x=tauA, ls='--', lw=fit_lw, color=colors[0])
    ax.axvline(x=tauB, ls='--', lw=fit_lw, color=colors[2])
    ax.axhline(y=0.2, ls='--', lw=fit_lw, color='k')
    ax.set(xlabel="$t$", ylabel="$F_s(t, k=k_{\max})$", xscale='log', yscale='linear')
    nice_legend(ax)

    # if Sk:
    #     ax_Sk = axes[current_axis - 1]
    #     ax_Sk.axvline(x=kmaxAA, ls='--', lw=2*fit_lw, color=colors[0])
    #     ax_Sk.axvline(x=kmaxBB, ls='--', lw=2*fit_lw, color=colors[2])


# Self-part of overlap function.
if Q:
    ax = axes[current_axis]
    current_axis += 1

    # Q
    # columns=step,Q(t, a=a_A)_speciesA,Q(t, a=a_B)_speciesB
    data = np.loadtxt(data_file_Fs)
    step = data[:, 0]
    Q0 = data[:, 1]
    Q1 = data[:, 2]
    t = step * dt

    # columns=tau_A,a_A,tau_B,a_B,
    with open(qty_file_Q) as f:
        header = f.readline()
        data = f.readline().split(" ")
        tauA = float(data[0]) * dt  # tau is given in steps
        tauB = float(data[2]) * dt
        a_A  = float(data[1])
        a_B  = float(data[3])

    nice_plot(ax, t, Q0, color=colors[0], label=r"A ($\tau$ = %.3g)" % tauA, ls='-')
    nice_plot(ax, t, Q1, color=colors[2], label=r"B ($\tau$ = %.3g)" % tauB, ls='-')

    # ax.axvline(x=tauA, ls='--', lw=fit_lw, color=colors[0])
    # ax.axvline(x=tauB, ls='--', lw=fit_lw, color=colors[2])
    # ax.axhline(y=0.2, ls='--', lw=fit_lw, color='k')
    ax.set(xlabel="$t$", ylabel="$Q_s(t; a)$", xscale='log', yscale='linear')
    nice_legend(ax)

if chi4_Qs:
    ax = axes[current_axis]
    current_axis += 1

    # columns=step,chi4_Qs(t, a=0.300)_speciesA,chi4_Qs(t, a=0.300)_speciesB
    data = np.loadtxt(data_file_chi4_Qs)
    step = data[:, 0]
    chi4_Qs_0 = data[:, 1]
    chi4_Qs_1 = data[:, 2]
    t = step * dt

    # columns=tau_A,a_A,tau_B,a_B,
    # with open(qty_file_Q) as f:
    #     header = f.readline()
    #     data = f.readline().split(" ")
    #     tauA = float(data[0]) * dt  # tau is given in steps
    #     tauB = float(data[2]) * dt
    #     a_A  = float(data[1])
    #     a_B  = float(data[3])

    nice_plot(ax, t, chi4_Qs_0, color=colors[0], label=r"A", ls='-')
    nice_plot(ax, t, chi4_Qs_1, color=colors[2], label=r"B", ls='-')

    ax.set(xlabel="$t$", ylabel="$\chi_{4}^{Q_s}(t; a)$", xscale='log', yscale='linear')
    nice_legend(ax)


if chi4_Fs:
    ax = axes[current_axis]
    current_axis += 1

    # columns=step,chi4_Fs(t, k=6.351)_speciesA,chi4_Fs(t, k=6.889)_speciesB
    data = np.loadtxt(data_file_chi4_Fs)
    step = data[:, 0]
    chi4_Fs_0 = data[:, 1]
    chi4_Fs_1 = data[:, 2]
    t = step * dt

    nice_plot(ax, t, chi4_Fs_0, color=colors[0], label=r"A", ls='-')
    nice_plot(ax, t, chi4_Fs_1, color=colors[2], label=r"B", ls='-')

    ax.set(xlabel="$t$", ylabel="$\chi_{4}^{F_s}(t; q_{\max})$", xscale='log', yscale='linear')
    nice_legend(ax)



plt.tight_layout()
plt.show()
