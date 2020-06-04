import hoomd
from hoomd import md
import gsd.hoomd
import numpy as np
import copy
from collections import namedtuple
from my_hoomd_utils import hoomd_snapshot_to_gsd_frame, dump_hoomd_snapshot_to_xyz
import os 

import logging

Transition = namedtuple('Transition', ['step', 'liquid_snap', 'ihs_snap'])

class IHSTransitionFinder:
    ihs_max_dr_threshold = 1e-5

    def __init__(self, initial_snapshot, steps_per_block, ihs_period, ihs_traj_file, liq_traj_file, log_file=None):
        self.steps_per_block = steps_per_block
        self.ihs_period = ihs_period
        self.num_minimizations = 0
        self.snapshot_start_block = initial_snapshot

        self.ihs_traj_file = ihs_traj_file
        self.liq_traj_file = liq_traj_file
        # Start from clean file (do not append to existing file).
        if os.path.exists(ihs_traj_file):
            os.remove(ihs_traj_file)
        if os.path.exists(liq_traj_file):
            os.remove(liq_traj_file)

        self.blocks_completed = 0

        initial_quenched_snapshot = self.quench(initial_snapshot)
        self.temp_ihs_snapshots = [initial_quenched_snapshot]
        # First temp snapshot gets appended during run.
        self.temp_snapshots = []

        if log_file:
            self.logger = logging.getLogger('ihs_debugger')
            self.logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)


    def find_transitions_next_block(self):
        # Integrate the system forward, i.e. obtain the entire liquid trajectory.
        self.evolve_until_end_of_block()

        # First, quench last snap of block (We already have IHS of first snap of block).
        quenched_snap = self.quench(self.temp_snapshots[-1])
        self.temp_ihs_snapshots.append(quenched_snap)


        # Detect transitions recursively. There is an off-by-one error here!!
        block_offset = self.blocks_completed * (self.steps_per_block - 1)     # This assumes ihs_period = 1!!!!
        transitions = self.find_transitions(self.temp_snapshots, self.temp_ihs_snapshots[0], self.temp_ihs_snapshots[1], block_offset, block_offset + len(self.temp_snapshots) - 1)

        for t in transitions:
            dump_hoomd_snapshot_to_xyz(t.ihs_snap, self.ihs_traj_file, step=t.step, write_mode="a")
            dump_hoomd_snapshot_to_xyz(t.liquid_snap, self.liq_traj_file, step=t.step, write_mode="a")
        
        # For block index offset.
        self.blocks_completed += 1

        # Remove all temp_snapshots from memory.
        self.snapshot_start_block = self.temp_snapshots[-1]
        self.temp_snapshots = []

        # Remove first IHS of block. Last IHS becomes becomes new first IHS.
        self.temp_ihs_snapshots = [self.temp_ihs_snapshots[-1]]


    def find_transitions(self, states, ihs_start, ihs_end, start_index, end_index):
        # self.logger.debug('entering with %d states, start, stop = [%d, %d]' % (len(states), start_index, end_index))
        # First base case: ihs at start and end of time series is the same, and we assume no transitions.
        if self.compare_ihs(ihs_start, ihs_end):
            # self.logger.debug("end points the same, returning []")
            return []
        # Other base case: ihs at start and end of time series is different, but the time series consists of only two points:
        # we located the transition. Return last index before transition along with the liquid snapshot and ihs.
        elif len(states) == 2:
            # self.logger.debug("found transition at %d" % start_index)
            transition = Transition(step=start_index, liquid_snap=states[0], ihs_snap=ihs_start)
            return [transition]
        # General case: calculate ihs at midpoint and separately find transitions in time series (start, mid) and (mid, end).
        else:
            this_mid_index = len(states) // 2
            ihs_mid = self.quench(states[this_mid_index])
            mid_index = this_mid_index + start_index

            # self.logger.debug("looking between [%d, %d] and [%d, %d]" % (start_index, mid_index, mid_index, end_index))
            return self.find_transitions(states[:this_mid_index+1], ihs_start, ihs_mid, start_index, mid_index) + self.find_transitions(states[this_mid_index:], ihs_mid, ihs_end, mid_index, end_index)


    def compare_ihs(self, ihs1, ihs2):
        # Unwrap positions
        pos1 = np.copy(ihs1.particles.position)
        pos2 = np.copy(ihs2.particles.position)
        pos1_old = np.copy(ihs1.particles.position)
        pos2_old = np.copy(ihs2.particles.position)
        im1 = ihs1.particles.image
        im2 = ihs2.particles.image
        box = np.array( [ihs1.box.scale().Lz, ihs1.box.scale().Ly, ihs1.box.scale().Lz] )

        for d in range(3):
            pos1[:, d] += im1[:, d] * box[d]
            pos2[:, d] += im2[:, d] * box[d]

        dr2 =  np.apply_along_axis(lambda pos : np.dot(pos, pos), 1, pos1 - pos2) 
        max_dr = np.sqrt(np.max(dr2))
        print("max_dr: %g" % max_dr)

        return max_dr < self.ihs_max_dr_threshold


    def save_temp_snapshot_callback(self, step):
        self.temp_snapshots.append( self.take_snapshot() )


    def take_snapshot(self):
        return self.system.take_snapshot(integrators=False, dtype='double')


    def initialize_system_from_snapshot(self, snapshot):
        raise NotImplementedError()

    def quench(self, snapshot):
        raise NotImplementedError()

    
    def evolve_until_end_of_block(self):
        raise NotImplementedError()
