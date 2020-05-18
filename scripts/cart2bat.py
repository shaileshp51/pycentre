#!/usr/bin/env python

import os
import re
import sys
import logging
import textwrap
import argparse
import numpy as np
import pytraj as pt

from datetime import datetime

import pycentre
import pycentre.subneigh as subneigh 
import pycentre.mmsystem as mm
import pycentre.utils_ as utls
from  pycentre import *


def is_directory(dir_string):
    val = os.path.isdir(dir_string)
    if not val:
        raise argparse.ArgumentTypeError(
            "`%s` is not a valid directory" % dir_string)
    return dir_string

class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(CustomArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg


class TrajinType:
    trajins = []
    slices = []
    # trajout = []

    def __repr__(self):
        str_val = """trajins {} slices {}
        """.format(self.trajins, self.slices)
        return str_val


class RangeAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(RangeAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        atoms = []
        uTmp = values[0].split(",")
        for tmp in uTmp:
            m2 = re.match(r"(\d+)-(\d+)", tmp.strip())
            try:
                if (m2 and (len(m2.group()) >= 1)):
                    for x in range(int(m2.group(1)), int(m2.group(2)) + 1):
                        atoms.append(x)
                else:
                    m3 = re.match(r"\d+", tmp.strip())
                    if (m3 is not None):
                        atoms.append(int(m3.group(0)))
                    else:
                        raise Exception("Unrecognised atom selection: " + tmp)
            except Exception:
                raise Exception("Unrecognised atom selection: " + tmp)
        setattr(namespace, self.dest, atoms)


class TrajinAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(TrajinAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        trj = TrajinType()
        # print("values:", values)
        for vals in values:
            cmd_vs = vals.split(',')
            if len(cmd_vs) < 1:
                raise Exception("""Invalid traj command format: 
                there must one or more trjin-file and corresponding optional frame-slice. 
                see help message for more details.""")
            else:
                # trj.trajout.append(cmd_vs[0])
                trajins = []
                slicesin = []
                for trj_vs in cmd_vs[0:]:
                    vs = trj_vs.split(':')
                    # print(len(vs))
                    if 0 < len(vs) < 5:
                        trajins.append(vs[0])
                        try:
                            slice_tr = [int(x) for x in vs[1:]]
                            if(len(slice_tr) > 0):
                                slicesin.append(slice_tr)
                        except Exception as error:
                            print(error)
                            raise
                    else:
                        raise Exception("""Invalid traj command format: 
                there must be trjin-file followed by maximum three int for frame-slice
                see help message for more details."""
                                        )
                trj.trajins.append(trajins)
                trj.slices.append(slicesin)
                if len(trajins) != len(slicesin) and len(slicesin) != 0:
                    raise Exception(
                        """Either all trajs must contain slices info or none""")
        setattr(namespace, self.dest, trj)





parser = CustomArgumentParser(fromfile_prefix_chars='@', add_help=False)
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument("-top",
                           "--topology",
                           help="topology file",
                           required=True)

requiredNamed.add_argument("-r",
                           "--root-atoms",
                           help="indices of 3-root atoms(1-based)",
                           type=int,
                           nargs=3,
                           required=True)
parser.add_argument("-mdprog",
                           "--md-program",
                           help="MD simulations which has generated trajectories",
                           choices={"amber", "charmm", "namd", "gromacs"},
                           default="amber")
parser.add_argument("-wd",
                    "--work-directory",
                    type=is_directory,
                    help="absolute path of working directory (default: current working directory)",
                    default=os.getcwd())
parser.add_argument("-atm",
                    "--atoms",
                    #nargs=1,
                    #action=RangeAction,
                    help="""selection string for atoms involving which bonds and
                     angles are considered for entropy calculation: as for cpptraj""",
                    )
parser.add_argument("-psb",
                    "--pseudo-bonds",
                    help="atom indices(1-based) of atoms involved in pseudo bonds: comma-separeted pair without spaces (default: None)",
                    type=int,
                    action="store",
                    nargs='*')
parser.add_argument("-nochunk",
                    "--no-chunk",
                    help="do not use chunk of frames from input trajectory for process (default: False)",
                    action="store_true")
parser.add_argument("-csz",
                    "--chunk-size",
                    help="chunk-size: number of frames in chunk (default: 4096)",
                    type=int,
                    default=4096)
parser.add_argument("-nph",
                    "--no-phase",
                    help="do not use pahse for torsions (default: use-phase)",
                    action="store_true")
parser.add_argument("-nrd",
                    "--no-radian",
                    help="do not use radian, use degree. (default: False, i.e. radian)",
                    action="store_true")
parser.add_argument("-ndfs",
                    "--no-dfs",
                    help="do not use depth-first-search for constructing mol. tree, (default: False i.e. dfs)",
                    action="store_true")
parser.add_argument("-phref",
                    "--phaseref-atoms",
                    help="selection-string for phase-reference atoms. (default: :*@N,CA,C)",
                    default=":*@N,CA,C")
parser.add_argument("-out",
                    "--output",
                    help="name of file to write stdout output (default: sys.stdout)",
                    default=sys.stdout)
parser.add_argument("-info",
                    "--information",
                    help="information about progress of run. (default: work-dir/info-cart2bat.dat)",
                    default="info-cart2bat.dat")
parser.add_argument("-tree",
                    "--tree-file",
                    help="filename for writing molecule-tree. (default: work-dir/tree-cart2bat.dat)",
                    default="tree-cart2bat.dat")
parser.add_argument('--version', action='version', version='%(prog)s version='+str(pycentre.__version__))


subsetgrp = parser.add_argument_group(
    'Subset-DOFs and NeighborDOFS command group')
subsetgrp.add_argument("-sub", "--subset-sel",
                       default=":*@H=",
                       help="selection-string for atoms, DOFs(bonds and angles) involving which are not-considered for entropy calculation: as for cpptraj (default: :*@H=)")
subsetgrp.add_argument("-cut", "--cutoff",
                       help="distance cutoff in angstrom for neighbor consideration (default: 14)",
                       type=int, action="store", default=[14], nargs='*')
subsetgrp.add_argument("-osf", "--out-subset-file",
                       help="filename for subset output (default: work-dir/subset.txt)", default="subset.txt")
subsetgrp.add_argument("-nneigh",
                       "--no-neighbor",
                       help="do not create neighbor list for default distance cutoff, (default: False)",
                       action="store_true")
subsetgrp.add_argument("-odf", "--out-distance-file",
                       help="filename for distance matrix (default: work-dir/distmatrix.txt)", default="distmatrix.txt")
subsetgrp.add_argument("-dskip", "--dist-frame-skip",
                       type=int, 
                       default=100,
                       help="consider every skip frame for avg. atom-atom distance calculation (default: 100)")
subsetgrp.add_argument("-onfp", "--out-neigh-fprefix",
                       help="filename prefix for neighbor output (default: neigh)",
                       default="neigh")
subsetgrp.add_argument("-bo", "--bond-order",
                       help="bond-order for entropy calculation (default: 2)",
                       type=int,
                       default=2)
subsetgrp.add_argument("-ao", "--angle-order",
                       help="angle-order for entropy calculation (default: 2)",
                       type=int,
                       default=2)
subsetgrp.add_argument("-to", "--torsion-order",
                       help="torsion-order for entropy calculation (default: 2)",
                       type=int,
                       default=2)
subsetgrp.add_argument("-nba", "--no-ba-cross",
                       help="include bond-angle-cross in entropy calculation (default: True, i.e. ba-cross)",
                       action="store_false")
subsetgrp.add_argument("-nbt", "--no-bt-cross",
                       help="include bond-torsion-cross in entropy calculation (default: True, i.e. bt-cross)",
                       action="store_false")
subsetgrp.add_argument("-nat", "--no-at-cross",
                       help="include bond-torsion-cross in entropy calculation (default: True, i.e. at-cross)",
                       action="store_false")

child_parser = CustomArgumentParser(fromfile_prefix_chars='@', parents=[parser],
                                    prog='cart2bat.py',
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    description=textwrap.dedent('''\
         Please do consider possibility given below as well!
         ----------------------------------------------------------------------------
             You can keep these command-line-argument(s) and their values in a file
             say (cmd_args.txt), each option in seperate-line, you can also keep
             comment lines starting with character '#' in the file, Then involke
             program to read it from file using:
             cart2bat.py @cmd_args.txt
         ****************************************************************************
         '''))
group = child_parser.add_argument_group('required traj command group')
group.add_argument('-trajin', '--trajin', nargs=1,
                   default=[],
                   action=TrajinAction,
                   required=True,
                   help="""{trajin-file:<first>:<last>:<skip>},+ 
                        i.e. one or more occurrences of argument-group: 
                        trajin-file:<first>:<last>:<skip>
                        <XX>: optional argument, but all occurrence must me consistent
                        i.e. either all the argument-groups has slice information or none""")
group.add_argument('-trajout', '--trajout', nargs=1,
                   default=None,
                   # action=store,
                   required=True,
                   help="""trajout-file, this file will be created in work-dir, if absolute path to filename is not provided.""")


# test_args_list = ['-mdprog', 'amber', '-top', 'abc.top',  '-atm', '1-1200', '-r',
#                                 '2', '1', '5', '-traj', 'trajout.nc,trajin1.nc:1:2:3,trajin2.nc:10:20',
#                                 '-traj', 'pqr.nc,trajin.nc']

# args.atoms = utilities.utils_.list_to_ranges(args.atoms)


def validate_args(args):
    abswd = utls.get_absfile(args.work_directory, os.getcwd(), 'r')
    # print(abswd)
    args.work_directory = abswd
    res = utls.is_validfile('-top', args.topology, args.work_directory, 'r')
    for trajin in args.trajin.trajins:
        for trj in trajin:
            utls.is_validfile('-trajin', trj, args.work_directory, 'r')
    res = utls.is_validfile('-trajout', args.trajout[0], args.work_directory, 'w')
    if args.output is not sys.stdout:
        res = utls.is_validfile('-out', args.output, args.work_directory, 'w')
    res = utls.is_validfile('-info', args.information, args.work_directory, 'w')
    res = utls.is_validfile('-tree', args.tree_file, args.work_directory, 'w')
    return res


# print()
# print(args)

def str_arguments(obj_args, sep='='):
    outputs = []
    if obj_args.md_program:
        outputs.append("{0:20s} {1} {2}".format(
            '--md-program', sep, str(obj_args.md_program)))
    if obj_args.atoms:
        outputs.append("{0:20s} {1} {2}".format(
            '--atoms', sep, obj_args.atoms))
    if obj_args.root_atoms:
        outputs.append("{0:20s} {1} {2}".format(
            '--root-atoms', sep, str(obj_args.root_atoms)))
    if obj_args.pseudo_bonds:
        outputs.append("{0:20s} {1} {2}".format(
            '--pseudo-bonds', sep, str(obj_args.pseudo_bonds)))
    if obj_args.no_phase:
        outputs.append("{0:20s} {1} {2}".format(
            '--no-phase', sep, str(obj_args.no_phase)))
    if obj_args.no_radian:
        outputs.append("{0:20s} {1} {2}".format(
            '--no-radian', sep, str(obj_args.no_radian)))
    if obj_args.no_dfs:
        outputs.append("{0:20s} {1} {2}".format(
            '--no-dfs', sep, str(obj_args.no_dfs)))
    if obj_args.no_chunk:
        outputs.append("{0:16s} {1} {2} frames".format(
            '--no-chunk', sep, str(obj_args.no_chunk)))
    if not obj_args.no_chunk:
        outputs.append("{0:20s} {1} {2} frames".format(
            '--chunk-size', sep, str(obj_args.chunk_size)))
    if obj_args.work_directory:
        outputs.append("{0:20s} {1} {2}".format(
            '--work-directory', sep, str(obj_args.work_directory)))
    if obj_args.output:
        tmp = 'sys.stdout' if obj_args.output is sys.stdout else str(
            obj_args.output)
        outputs.append("{0:20s} {1} {2}".format('--output', sep, tmp))
    if obj_args.tree_file:
        outputs.append("{0:20s} {1} {2}".format(
            '--tree-file', sep, obj_args.tree_file))
    if obj_args.information:
        outputs.append("{0:20s} {1} {2}".format(
            '--information', sep, obj_args.information))
    if obj_args.topology:
        outputs.append("{0:20s} {1} {2}".format(
            '--topology', sep, str(obj_args.topology)))
    if obj_args.phaseref_atoms:
        outputs.append("{0:20s} {1} {2}".format(
            '--phaseref-atoms', sep, str(obj_args.phaseref_atoms)))        

    if obj_args.subset_sel:
        outputs.append("{0:20s} {1} {2}".format(
            '--subset-sel', sep, str(obj_args.subset_sel)))     
    if obj_args.out_subset_file:
        outputs.append("{0:20s} {1} {2}".format(
            '--out-subset-file', sep, str(obj_args.out_subset_file)))  
    if not obj_args.no_neighbor:
        outputs.append("{0:20s} {1} {2}".format(
        '--no-neighbor', sep, str(obj_args.no_neighbor)))
        if len(obj_args.cutoff) > 0:
            outputs.append("{0:20s} {1} {2}".format(
                '--cutoff', sep, ",".join([str(c) for c in obj_args.cutoff]))) 
        if obj_args.out_distance_file:
            outputs.append("{0:20s} {1} {2}".format(
            '--out-distance-file', sep, str(obj_args.out_distance_file)))
        if obj_args.dist_frame_skip:
            outputs.append("{0:20s} {1} {2}".format(
            '--dist-frame-skip', sep, str(obj_args.dist_frame_skip)))
        if obj_args.out_neigh_fprefix:
            outputs.append("{0:20s} {1} {2}".format(
            '--out-neigh-fprefix', sep, str(obj_args.out_neigh_fprefix)))
        if obj_args.bond_order:
            outputs.append("{0:20s} {1} {2}".format(
            '--bond-order', sep, str(obj_args.bond_order)))
        if obj_args.angle_order:
            outputs.append("{0:20s} {1} {2}".format(
            '--angle-order', sep, str(obj_args.angle_order)))
        if obj_args.torsion_order:
            outputs.append("{0:20s} {1} {2}".format(
            '--torsion-order', sep, str(obj_args.torsion_order)))
        if obj_args.no_ba_cross:
            outputs.append("{0:20s} {1} {2}".format(
            '--no-ba-cross', sep, str(obj_args.no_ba_cross)))
        if obj_args.no_bt_cross:
            outputs.append("{0:20s} {1} {2}".format(
            '--no-bt-cross', sep, str(obj_args.no_bt_cross)))
        if obj_args.no_at_cross:
            outputs.append("{0:20s} {1} {2}".format(
            '--no-at-cross', sep, str(obj_args.no_at_cross)))
                    
    if obj_args.trajin:
        for i, trajin in enumerate(obj_args.trajin.trajins):
            tmptrjs = []
            for j, trj in enumerate(trajin):
                tmp = []
                tmp.append(trj)
                if obj_args.trajin.slices[i] and obj_args.trajin.slices[i][j]:
                    tmp.extend(obj_args.trajin.slices[i][j])
                tmptrjs.append(':'.join([str(xx) for xx in tmp]))
            outputs.append("{0:20s} {1} {2}".format(
                '--trajin', sep, ','.join([str(xx) for xx in tmptrjs])))

    if obj_args.trajout:
        outputs.append("{0:20s} {1} {2}".format(
            '--trajout', sep, obj_args.trajout[0]))

    return ('\n'.join(outputs))


def timedelta2string(duration, rounding=3):
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

def main():
    args = child_parser.parse_args()
    is_argsvalid = validate_args(args)
    if is_argsvalid:
        logging.basicConfig(filename='apps.log', level=logging.DEBUG)
        fOut = sys.stdout
        if args.output is not sys.stdout:
            args.output = utls.get_absfile(args.output, args.work_directory, 'w')
            # print(args.output)
            fOut = open(args.output, 'w')
        #print("Cartesian-coordinate MD-trajectory to Bond-Angle-Torsion conversion version: %s\n" % __version__)
#         fOut.write(
#             "Cartesian-coordinate MD-trajectory Bond-Angle-Torsion conversion version: %s\n" % __version__)
        starttime = datetime.now()
        print("Process started on: %s\n" % starttime)
        fOut.write("Process started on: %s\n" % starttime)
        print(str_arguments(args))
        fOut.write(str_arguments(args) + "\n")
        abstop = utls.get_absfile(args.topology, args.work_directory, 'r')
        absinfo = utls.get_absfile(args.information, args.work_directory, 'w')
        args.topology = abstop
        args.information = absinfo
        if(args.md_program):
            sel_filter = args.subset_sel if args.subset_sel else args.atoms
            mol = mm.MMSystem(sel_build=args.atoms, sel_phaseref=args.phaseref_atoms, sel_filter=sel_filter)
            if args.md_program == "namd":
                mol = mm.NAMDSystem(sel_build=args.atoms, sel_phaseref=args.phaseref_atoms, sel_filter=sel_filter)
                mol.load_topology(args.topology)
            elif args.md_program == "amber":
                mol = mm.AMBERSystem(sel_build=args.atoms, sel_phaseref=args.phaseref_atoms, sel_filter=sel_filter)
                mol.load_topology(args.topology)
            elif args.md_program == "charmm":
                mol = mm.CHARMMSystem(sel_build=args.atoms, sel_phaseref=args.phaseref_atoms, sel_filter=sel_filter)
                mol.load_topology(args.topology)
            if mol is not None:
                pseudos = []
                if args.pseudo_bonds:
                    pseudos = list(args.pseudo_bonds)
                    # print(pseudos)
                (tree) = mol.create_bat_tree(
                    not args.no_dfs,
                    args.root_atoms,
                    args.no_phase,
                    pseudos)
                phasedef = []
                if(not args.no_phase):
                    phasedef = mol.assign_phase()
                    fOut.write("\nPhase Def (len) value : %d\n" %
                               len(phasedef))
                    bl_size = 10
                    for bl_from in range(0, len(phasedef), bl_size):
                        bl_to = len(phasedef) if len(
                            phasedef) < bl_from + bl_size else bl_from + bl_size
                        fOut.write("".join('%6d' %
                                           e for e in phasedef[bl_from:bl_to]) + '\n')
                abstree = utls.get_absfile(args.tree_file, args.work_directory, 'w')
                fOut.write('\nCheck molecule tree in file: %s\n' % abstree)
                with open(abstree, 'w') as fTree:
                    fTree.write(str(mol) + "\n")
                    
                if not args.trajin.trajins:
                    raise Exception("No Trajin command in input file")

                n_trjins = len(args.trajin.trajins)
                n_molatoms = mol.num_atoms()
                dist_mat = np.reshape(
                    np.zeros(n_molatoms * n_molatoms, dtype=np.float32), (n_molatoms, n_molatoms))
                for trj_i, trajin in enumerate(args.trajin.trajins):
                    proctrjs = []
                    for trj_j, trj in enumerate(trajin):
                        tmps = []
                        trjabs = utls.get_absfile(trj, args.work_directory, 'r')
                        tmps.append(trjabs)
                        if args.trajin.slices[trj_i] and args.trajin.slices[trj_i][trj_j]:
                            tmps.extend(args.trajin.slices[trj_i][trj_j])
                        proctrjs.append(tmps)

                    trajoutfile = utls.get_absfile(
                        args.trajout[0], args.work_directory, 'w')
                    arg = {'phasedef': phasedef,
                           'trajoutfile': trajoutfile, 'tree': tree}
                    logging.info(arg)
                    tmpouts = textwrap.dedent('''
                            ***processing (check --infomation file for progress of work)...
                                trajin  = %s
                                trajout = %s
                            ''' % (",".join([str(i1) for i1 in proctrjs]), trajoutfile))
                    print(tmpouts)
                    fOut.write(tmpouts)
                    fOut.flush()
                    logging.info(tmpouts)
                    if trj_i:
                        cart2bat_ = core.Cartesian2BAT(args.topology,
                                                       proctrjs,
                                                       trajoutfile,
                                                       True,
                                                       args.information,
                                                       args.no_chunk,
                                                       args.chunk_size,
                                                       args.root_atoms,
                                                       sorted(mol.atoms.keys()),
                                                       pseudos,
                                                       args.no_radian,
                                                       args.no_phase,
                                                       tree,
                                                       phasedef)
                    else:
                        cart2bat_ = core.Cartesian2BAT(args.topology,
                                                       proctrjs,
                                                       trajoutfile,
                                                       False,
                                                       args.information,
                                                       args.no_chunk,
                                                       args.chunk_size,
                                                       args.root_atoms,
                                                       sorted(mol.atoms.keys()),
                                                       pseudos,
                                                       args.no_radian,
                                                       args.no_phase,
                                                       tree,
                                                       phasedef)

                    msg = cart2bat_.convert_using_pytraj(proctrjs)
                    logging.info(msg)
                    fOut.flush()
                    if not args.no_neighbor:
                        # absout_distance_file = utls.get_absfile(args.out_distance_file, args.work_directory, 'w')
                        fOut.write("Calculating distance matrix, it can be time takeing\n")
                        fOut.flush()
                        dist_mat2 = cart2bat_.distance_matrix(
                            proctrjs, args.atoms, args.dist_frame_skip)
                        fOut.write("Distances calculated, updating distance matrix\n")
                        fOut.flush()
                        dist_mat += dist_mat2
                        # np.savetxt(absout_distance_file, dist_mat, fmt='%-7.2f')
                        
                if not args.no_neighbor:
                    dist_mat /= n_trjins
                    absout_distance_file = utls.get_absfile(args.out_distance_file, args.work_directory, 'w')
                    np.savetxt(absout_distance_file, dist_mat, fmt='%-7.2f')
                    print(
                        "Getting: indices of selected atoms for subset determination...")
                    mytop = pt.load_topology(args.topology)
                    selindices = mytop.select(args.subset_sel)
                    print("Reading: Mol. System Tree..")
                    tree.load(
                        abstree, header=True, sel_filter=args.subset_sel)
                    moltree = tree.to_dataframe()
                    print("Writing: subset input file for entropy calculation...")
                    absout_subset_file = utls.get_absfile(args.out_subset_file, args.work_directory, 'w')
                    subset_dofs = subneigh.SubsetDOFs(sel_filter=args.subset_sel)
                    subset_dofs.set_moltree(moltree)
                    subset_dofs.write(absout_subset_file)
                    n_bnd_eff, n_ang_eff, n_tor_eff = subset_dofs.n_bnd_eff, subset_dofs.n_ang_eff, subset_dofs.n_tor_eff
#                     n_bnd_eff, n_ang_eff, n_tor_eff = subneigh.write_subsetfile(
#                         moltree, absout_subset_file)
                    print("eff-bonds: {}, eff-angles: {}, torsions: {}".format(n_bnd_eff,
                                                                               n_ang_eff, n_tor_eff))
                    fOut.write("eff-bonds: {}, eff-angles: {}, torsions: {}".format(n_bnd_eff,
                                                                               n_ang_eff, n_tor_eff))
                    for cu in args.cutoff:
                        print("Building: neighbor-list input file with distance cutoff(" +
                              str(cu) + ") for entropy calc...")
                        neighbors = subneigh.neighbor_builder(dist_mat, 
                                                              tree, 
                                                              subset_dofs,
                                                              selindices,
                                                              args.bond_order == 2, 
                                                              args.angle_order == 2, 
                                                              args.torsion_order == 2,
                                                              args.no_ba_cross, 
                                                              args.no_bt_cross, 
                                                              args.no_at_cross, 
                                                              cuttoff=cu)
                        print("Writing: neighbor-list input file with distance cutoff(" +
                              str(cu) + ") for entropy calc...")
                        absout_neigh_fprefix = utls.get_absfile(args.out_neigh_fprefix, 
                                                           args.work_directory, 
                                                           'w')
                        msgsn = neighbors.write(absout_neigh_fprefix + "_" + str(cu) + ".txt")
                        fOut.write('\n\n' + msgsn)
                        fOut.flush()
        endtime = datetime.now()
        elapsed_time = endtime - starttime
        print("\n\nProcess finished on: %s, Time Taken: %s\n" % (endtime, timedelta2string(elapsed_time)))
        fOut.write("\n\nProcess finished on: %s, Time Taken: %s\n" % (endtime, timedelta2string(elapsed_time)))
        fOut.close()


if __name__ == "__main__":
    main()
