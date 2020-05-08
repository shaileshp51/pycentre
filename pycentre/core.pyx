#!/usr/bin/env python
# cython: profile=True

import logging
from datetime import datetime
from datetime import timedelta
logger = logging.getLogger(__name__)


import numpy as np
import pytraj as pt
from math import pi as PI
from datetime import datetime


try:
    import batio as trajIO
except ImportError:
    try:
        from . import batio as trajIO
    except ImportError as err:
        raise ("Error import: " + str(err))


class ProgressState():
    def __init__(self, total):
        self.total = total
        self.starttime = datetime.now()
        self.updatetime = self.starttime
        self.done = 0

    def update(self, incr_done):
        self.done += incr_done
        self.updatetime = datetime.now()
        if self.done > self.total:
            self.done = self.done

    def timedelta2string(self, duration, rounding=3):
        seconds = duration.total_seconds()
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        second = (seconds % 60)
        mm = '0' + str(minutes) if minutes < 10 else str(minutes)
        ss = '0' + str(round(second, rounding)
                       ) if second < 10 else str(round(second, 3))
        result = ''
        if hours > 0 and minutes > 0:
            result = '{0}h:{1}m:{2}s'.format(int(hours), mm, ss)
        elif minutes > 0:
            result = '{0}m:{1}s'.format(mm, ss)
        else:
            result = '{0}s'.format(ss)
        return result

    def progress(self):
        pp = self.done * 100.0 / self.total if self.done > 0 else 0.0
        elapsed = self.updatetime - self.starttime
        elapsed_sec = elapsed.total_seconds()
        avg_sec = elapsed_sec / self.done if self.done > 0 else 0.0
        avg_mus = elapsed_sec * 1000000 / self.done if self.done > 0 else 0.0
        avgtasktime = timedelta(
            seconds=avg_sec, microseconds=avg_mus) if self.done > 0 else 0.0
        ppleft = (100.0 - pp) * elapsed_sec / pp if self.done > 0 else 0.0
        estimate = timedelta(seconds=ppleft) if self.done > 0 else 0.0
        output = 'Total %s frames to process, progress is not known yet.' % self.total
        if self.done > 0:
            output = 'Processed: [%s/%s] [%s %%] in [%s], avgerage-time/frame [%s]  estimated-time-left [%s]' % (self.done,
                                                                                                                 self.total, round(pp, 2), self.timedelta2string(
                                                                                                                     elapsed), self.timedelta2string(avgtasktime, 6),
                                                                                                                 self.timedelta2string(estimate))
        return output


class Cartesian2BAT():
    def __init__(self, topology, trajins, trajout, append_trajout, infofile,
                 no_chunk, chunk_size, root_atoms, atoms, pseudo_bonds,
                 no_radian, no_phase, tree, phaseDef):
        self.topology = topology
        self.trajins = trajins
        self.trajout = trajout
        self.append_trajout = append_trajout
        self.infofile = infofile
        self.no_chunk = no_chunk
        self.chunk_size = chunk_size
        self.atoms = atoms
        self.root_atoms = root_atoms
        self.pseudo_bonds = pseudo_bonds
        self.no_radian = no_radian
        self.no_phase = no_phase
        self.tree = tree
        self.phase_defn = phaseDef

    # @profile
    def deg2rad_n_phase_correction(self, angles_val, dihedrals_val):
        cdef int i, j, k, fNo
        cdef double modFactor
        cdef double deg2rad, _2PI
        if not self.no_radian:
            deg2rad = PI / 180.0
            _2PI = 2 * PI
            angles_val = angles_val * deg2rad
            dihedrals_val = dihedrals_val * deg2rad
            # print(dihedrals_val.shape, type(dihedrals_val))
            # move range (-PI, PI) -> (0.0, 2*PI) by adding 2*PI
#             B = np.copy(angles_val)
#             C = np.copy(dihedrals_val)
#             for i in range(B.shape[0]):
#                 for j in range(B.shape[1]):
#                     if B[i, j] < 0.0:
#                         B[i, j] += _2PI
#             for i in range(C.shape[0]):
#                 for j in range(C.shape[1]):
#                     if C[i, j] < 0.0:
#                         C[i, j] += _2PI
            angles_val[angles_val < 0.0] += _2PI
            dihedrals_val[dihedrals_val < 0.0] += _2PI
            #print("angles equal: ", np.array_equal(angles_val, B))
            #print("dih equal", np.array_equal(dihedrals_val, C))
        if not self.no_phase:
            # Substract value of phase angle if phase
            # if modified torsion becomes negative add 2*PI [rad] or 180 [deg]
            # if modified torsion is positive substract PI and in deg substract
            # 180
            modFactor = 360.0 if self.no_radian else _2PI
            for k in range(0, dihedrals_val.shape[0]):
                if (k != self.phase_defn[k + 1] - 1):
                    for fNo in range(dihedrals_val.shape[1]):
                        dihedrals_val[k,
                                      fNo] -= dihedrals_val[self.phase_defn[k + 1] - 1, fNo]
                        if (dihedrals_val[k, fNo] < 0.0):
                            dihedrals_val[k, fNo] += modFactor
                        if (self.no_radian):
                            dihedrals_val[k, fNo] -= 180.0

        return angles_val, dihedrals_val

    # @profile
    def convert_using_pytraj(self, trajIn):
        bonds_list = []
        angles_list = []
        torsions_list = []

        for k in sorted(self.tree.nodes.keys()):
            if self.tree.nodes[k].a2 > 0:
                bonds_list.append([k - 1, self.tree.nodes[k].a2 - 1])
            if self.tree.nodes[k].a2 > 0 and self.tree.nodes[k].a3 > 0:
                angles_list.append(
                    [k - 1, self.tree.nodes[k].a2 - 1, self.tree.nodes[k].a3 - 1])
            if self.tree.nodes[k].a2 > 0 and self.tree.nodes[k].a3 > 0 and self.tree.nodes[k].a4 > 0:
                torsions_list.append(
                    [k - 1, self.tree.nodes[k].a2 - 1, self.tree.nodes[k].a3 - 1, self.tree.nodes[k].a4 - 1])

        n_atom = len(self.atoms)
        pseudo_bonds = None
        if len(self.pseudo_bonds) % 2 == 0:
            v_tmp = []
            for i in range(0, len(self.pseudo_bonds), 2):
                v_tmp.append((self.pseudo_bonds[i], self.pseudo_bonds[i + 1]))
            if len(v_tmp) > 0:
                pseudo_bonds = list(v_tmp)
        logger.debug('bond_indices: %s\nangle_indices: %s\n dih_indices%s' % (
            bonds_list, angles_list, torsions_list))
        logger.debug('pseudo_bonds: %s' % str(pseudo_bonds))
        logger.debug(
            str((n_atom, n_atom - 1, n_atom - 2, n_atom - 3, self.root_atoms)))
        trjs = []
        slices = []

        for tr1 in trajIn:
            trjs.append(tr1[0])
            if len(tr1) == 2:
                slices.append(tuple([0, tr1[1], 1]))
            elif len(tr1) == 3:
                slices.append(tuple([tr1[1] - 1, tr1[2], 1]))
            elif len(tr1) == 4:
                slices.append(tuple([tr1[1] - 1, tr1[2], tr1[3]]))
        if slices:
            if len(slices) == len(trjs):
                traj = pt.iterload(trjs, self.topology, frame_slice=slices)
            else:
                raise Exception("Either all trajin should have slices or none")
        else:
            traj = pt.iterload(trjs, self.topology)
        traj_frames_eff = traj.n_frames
        if self.no_chunk:
            block_size = traj.n_frames
        else:
            block_size = self.chunk_size if self.chunk_size < traj.n_frames else traj.n_frames
        print(str(("Total number of frames: %d, chunksize: %d" %
                   (traj.n_frames, block_size))))
        logger.debug(
            str(("Total number of frames: %d, chunksize: %d" % (traj.n_frames, block_size))))
        chunk_id = 0
        print(self.infofile)
        #progressstate = {'total': traj_frames_eff, 'chunk_size': block_size, 'done': 0}
        progressstate = ProgressState(traj_frames_eff)
        for chunk in traj.iterchunk(block_size, start=0, stop=-1):
            chunk_start = chunk_id * block_size
            chunk_end = (chunk_id + 1) * block_size if (chunk_id + 1) * \
                block_size < traj_frames_eff else traj_frames_eff
            #print("block stast, end: ", chunk_id * block_size, chunk_end)
            if chunk_end - chunk_id * block_size > 0:
                logger.debug('Processing %s block %i Frames %i to %i' % (
                    str(trajIn), chunk_id + 1, chunk_start, chunk_end))
                bonds_val = pt.distance(chunk, bonds_list, dtype='ndarray')
                angles_val = pt.angle(chunk, angles_list, dtype='ndarray')
                dihedrals_val = pt.dihedral(
                    chunk, torsions_list, dtype='ndarray')
#                 for i in range(200):
#                     self.deg2rad_n_phase_correction(angles_val, dihedrals_val, self.inputs['useDegree'], self.inputs['usePhase'])
                angles_val, dihedrals_val = self.deg2rad_n_phase_correction(
                    angles_val, dihedrals_val)

                logger.debug(
                    str((bonds_val.shape, type(bonds_val), bonds_val)))
                if chunk_id == 0:
                    logger.debug("trying to create nc file for bonds")
                    trjBAT = trajIO.ncbat(self.trajout, n_atom, n_atom - 1, n_atom - 2,
                                          n_atom - 3, self.root_atoms, pseudo_bonds=pseudo_bonds)
                    if not self.append_trajout:
                        trjBAT.create_dataset(
                            "NetcdfBAT created using cart2bat")

                frames_indices = [i for i in range(
                    chunk_start + 1, chunk_end + 1)]
                logger.debug("Writing BAT trajectory...\n")
                t_bonds_val = np.transpose(bonds_val)
                t_angles_val = np.transpose(angles_val)
                t_dihedrals_val = np.transpose(dihedrals_val)
                frm_o, frm_n = trjBAT.append_frames(
                    frames_indices, t_bonds_val, t_angles_val, t_dihedrals_val)
                logger.debug(
                    str("%d frames appended to file successfully\n" % (frm_n - frm_o)))
            else:
                logger.critical("Exception: There are no frames to process..")
                raise Exception("There are no frames to process..")
            chunk_id += 1
            progressstate.update(chunk_end - chunk_start)
            with open(self.infofile, 'w') as fpinfo:
                fpinfo.write(progressstate.progress())
        logger.debug("Processing input trajectory successful...")
        return(True)

    # @profile
    def distance_matrix(self, trajIn, atomsel, skip=100):
        trjs = []
        slices = []
        
        for tr1 in trajIn:
            trjs.append(tr1[0])
            if len(tr1) == 2:
                slices.append(tuple([0, tr1[1], 1 * skip]))
            elif len(tr1) == 3:
                slices.append(tuple([tr1[1] - 1, tr1[2], 1 * skip]))
            elif len(tr1) == 4:
                slices.append(tuple([tr1[1] - 1, tr1[2], tr1[3] * skip]))
        if slices:
            if len(slices) == len(trjs):
                traj = pt.iterload(trjs, self.topology, frame_slice=slices)
            else:
                raise Exception("Either all trajin should have slices or none")
        else:
            traj = pt.iterload(trjs, self.topology, stride=skip)
        traj_frames_eff = traj.n_frames
        dist_mat = pt.distance_matrix(traj, mask=atomsel)
        
        return dist_mat
        
