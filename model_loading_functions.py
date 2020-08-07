import numpy as np
import os 
from lerner_group import sim_lib as sim
from lerner_group.init_model import InitializeModel
from lerner_group.read_tools import get_config_snap_corrado,get_config_snap_edan,get_config_snap_misaki,get_config_snap_sylvain





def get_sim_edan(N, serial, delta=0, cooling_rate=1e-3):
    file_name = "slowly_quenched_solid2D_N%d_%04d.dat" % (N, serial)
    folder = "%s/ipl/slowlyQuenchedSolids/2D/%d/" % (os.getenv("MD_DATA_DIR"), N)
    file_name = os.path.join(folder, file_name)
    npart, pos, ptypes, box, strain = get_config_snap_edan(file_name)
    box_size = np.array([box, box])
    diam = np.zeros(npart)
    diam[ptypes == 0] = 1.
    diam[ptypes == 1] = 1.4
    pos *= box

    sim = InitializeModel(model_name='edan', r=pos, diameter=diam,
                          box_size=box_size, strain=strain).sim
    print('u/N =', sim.pairwise.compute_pot()/npart)
    sim.pairwise.compute_everything(delta)
    print("typGrad/p = %g" % (sim.system.typical_grad / sim.system.thermo_pre))

    return sim, npart, pos, diam, ptypes, box, strain


def get_sim_ipl10(N, T, serial, delta=0):
    file_name = "%s/swap/solids/2D/%d/%g/swap2D_N%d_T%g_%d.dat" % (os.getenv("MD_DATA_DIR"), N, T, N, T, serial)
    npart,pos,diam,box,strain,dummy = get_config_snap_corrado(file_name)
    box_size  = np.array([box,box])
    pos *= box

    sim = InitializeModel(model_name='ipl10', r=pos, diameter=diam, box_size=box_size, strain=strain).sim

    print('u/N =',sim.pairwise.compute_poly_pot()/npart)
    sim.pairwise.compute_poly_everything()
    print("ratio = %g" % (sim.system.typical_grad/sim.system.thermo_pre))

    return sim, npart, pos, diam, box, strain
