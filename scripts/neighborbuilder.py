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

 
    
def main():
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument(
        "-top", "--topology", help="file: pdb format, of molecular system", required=True)
    requiredNamed.add_argument(
        "-t", "--tree", help="tree-file of molecular system", required=True)
    parser.add_argument(
        "-s", "--sel", help="sel_fliter string for atoms involving which bonds and angles are not-considered for entropy calculation: as for cpptraj (default: :*@H=)", default=":*@H=")
    parser.add_argument("-c", "--cutoff", help="distance cutoff in angstrom for neighbor consideration (default: 14)",
                        type=int, action="store", default=[14], nargs='*')
    parser.add_argument("-osf", "--out-subset-file",
                        help="filename for subset output (default: workdir/subset.txt)", default="subset.txt")
    parser.add_argument(
        "-d", "--dist", help="atom-atom distance file molecular system (default: workdir/distmatrix.txt)")
    parser.add_argument("-onfp", "--out-neighbor-file-prefix",
                        help="filename prefix for neighbor output (default: neigh)", default="neigh")
    parser.add_argument("-bo", "--bond-order",
                        help="bond-order for entropy calculation (default: 2)", type=int, default=2)
    parser.add_argument("-ao", "--angle-order",
                        help="angle-order for entropy calculation (default: 2)", type=int, default=2)
    parser.add_argument("-to", "--torsion-order",
                        help="torsion-order for entropy calculation (default: 2)", type=int, default=2)
    parser.add_argument("-nba", "--no-ba-cross",
                        help="include bond-angle-cross in entropy calculation (default: True, i.e. ba-cross)", action="store_false")
    parser.add_argument("-nbt", "--no-bt-cross",
                        help="include bond-torsion-cross in entropy calculation (default: True, i.e. ba-cross)", action="store_false")
    parser.add_argument("-nat", "--no-at-cross",
                        help="include bond-torsion-cross in entropy calculation (default: True, i.e. ba-cross)", action="store_false")
    parser.add_argument('--version', action='version', version='%(prog)s version='+str(pycentre.__version__))
    args = parser.parse_args()

    print("Reading: Mol. System Tree..")
    tree = mmsystem.MMTree()
    tree.load(args.tree, header=True, sel_filter=args.sel)
    moltree = tree.to_dataframe()
    # moltree = read_tree_as_dataframe(args.tree, args.pdb, args.sel)
    print("Getting: indices of selected atoms for subset determination...")
    
    mytop = pt.load_topology(args.topology)
    selindices = mytop.select(args.sel)
    print("Reading: Mol. System Tree..")
    tree.load(
        args.tree, header=True, sel_filter=args.sel)
    moltree = tree.to_dataframe()
    print("Writing: subset input file for entropy calculation...")
    # absout_subset_file = utls.get_absfile(args.out_subset_file, args.work_directory, 'w')
    subset_dofs = subneigh.SubsetDOFs(sel_filter=args.sel)
    subset_dofs.set_moltree(moltree)
    print("Writing: subset input file for entropy calculation...")
    subset_dofs.write(args.out_subset_file)
    n_bnd_eff, n_ang_eff, n_tor_eff = subset_dofs.n_bnd_eff, subset_dofs.n_ang_eff, subset_dofs.n_tor_eff

    print("eff-bonds: {}, eff-angles: {}, torsions: {}".format(n_bnd_eff,
                                                            n_ang_eff, n_tor_eff))
    if args.dist:
        print("Loading: Atom-atom distance matrix...")
        distances = np.loadtxt(args.dist)
        for cu in args.cutoff:
            print("Building: neighbor-list input file with distance cutoff(" +
               str(cu) + ") for entropy calc...")
            neighbors = subneigh.neighbor_builder(distances, tree, subset_dofs, selindices,
                                      args.bond_order == 2, args.angle_order == 2, args.torsion_order == 2,
                                      args.no_ba_cross, args.no_bt_cross, args.no_at_cross, cuttoff=cu)
            print("Writing: neighbor-list input file with distance cutoff(" +
               str(cu) + ") for entropy calc...")
            neighbors.write(args.out_neighbor_file_prefix + "_" + str(cu) + ".txt")


if __name__ == "__main__":
    main()
