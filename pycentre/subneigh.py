#!/usr/bin/env python3

import numpy as np
import pytraj as pt
import pandas as pd
import copy
import argparse

try:
    from pycentre import mmsystem
    from pycentre.subsetdofs import *
    from pycentre.neighbordofs import *
except:
    pass
    #import mmsystem


def get_atom_pairs(t1, t2, allow_xx=False):
    tups = []
    for i in t1:
        for j in t2:
            if ((i, j) not in tups) and ((j, i) not in tups):
                if(i == j and allow_xx):
                    tups.append((i, j))
                elif(i != j):
                    tups.append((i, j))
    return tups


def is_neighbor(distmat, t1, t2, allow_xx, cuttoff):
    tups = get_atom_pairs(t1, t2, allow_xx)
    res = False
    n_tup = len(tups)
    tup_i = 0
    while tup_i < n_tup and (not res):
        if distmat[tups[tup_i][0] - 1][tups[tup_i][1] - 1] <= cuttoff:
            res = True
        tup_i += 1
    return res


def neighbor_builder(distmat, moltree, subset_dofs, sel_indices, bb, aa, tt, ba, bt, at, cuttoff=6.0):
    if isinstance(moltree, mmsystem.MMTree):
        bonds, angles, torsions = {}, {}, {}
        bid, aid, did = 0, 0, 0
        for kn in sorted(moltree.nodes.keys()):
            node = moltree.nodes[kn]
            a1, a2, a3, a4 = node.a1, node.a2, node.a3, node.a4
            if (a1 != -1) and (a2 != -1):
                bonds[bid] = (a1, a2)
                bid += 1
            if (a1 != -1) and (a2 != -1) and (a3 != -1):
                angles[aid] = (a1, a2, a3)
                aid += 1
            if (a1 != -1) and (a2 != -1) and (a3 != -1) and (a4 != -1):
                torsions[did] = (a1, a2, a3, a4)
                did += 1
        n_bnd, n_ang, n_tor = len(bonds), len(angles), len(torsions)
        resl = NeighborDOFs()
        resl.set_subset(subset_dofs)
        bb_idx, aa_idx, tt_idx, ba_idx, bt_idx, at_idx = 0, 0, 0, 0, 0, 0
        if bb or ba or bt:
            for b1 in range(n_bnd):
                b1_ni_sel = False
                b1_ni_sel = (
                    bonds[b1][0] - 1 not in sel_indices) and (bonds[b1][1] - 1 not in sel_indices)
                if bb and b1_ni_sel:
                    neigh_list = []
                    for b2 in range(b1 + 1, n_bnd):
                        b2_ni_sel = False
                        b2_ni_sel = (
                            bonds[b2][0] - 1 not in sel_indices) and (bonds[b2][1] - 1 not in sel_indices)
                        if b2_ni_sel and is_neighbor(distmat, bonds[b1], bonds[b2], allow_xx=False, cuttoff=cuttoff):
                            neigh_list.append(b2)
                    if len(neigh_list) > 0:
                        resl.bb_ids_[b1] = {
                            v: bb_idx + k for k, v in enumerate(neigh_list)}
                        bb_idx += len(neigh_list)
                if ba and b1_ni_sel:
                    neigh_list = []
                    for a2 in range(n_ang):
                        a2_ni_sel = False
                        a2_ni_sel = (
                            angles[a2][0] - 1 not in sel_indices) and (angles[a2][1] - 1 not in sel_indices)
                        a2_ni_sel = a2_ni_sel and (
                            angles[a2][2] - 1 not in sel_indices)
                        if a2_ni_sel and is_neighbor(distmat, bonds[b1], angles[a2], allow_xx=False, cuttoff=cuttoff):
                            neigh_list.append(a2)
                    if len(neigh_list) > 0:
                        resl.ba_ids_[b1] = {
                            v: ba_idx + k for k, v in enumerate(neigh_list)}
                        ba_idx += len(neigh_list)
                if bt and b1_ni_sel:
                    neigh_list = []
                    for t2 in range(n_tor):
                        if is_neighbor(distmat, bonds[b1], torsions[t2], allow_xx=False, cuttoff=cuttoff):
                            neigh_list.append(t2)
                    if len(neigh_list) > 0:
                        resl.bt_ids_[b1] = {
                            v: bt_idx + k for k, v in enumerate(neigh_list)}
                        bt_idx += len(neigh_list)
        if aa or at:
            for a1 in range(n_ang):
                a1_ni_sel = False
                a1_ni_sel = (
                    angles[a1][0] - 1 not in sel_indices) and (angles[a1][1] - 1 not in sel_indices) and (angles[a1][2] - 1 not in sel_indices)
                if aa and a1_ni_sel:
                    neigh_list = []
                    for a2 in range(a1 + 1, n_ang):
                        a2_ni_sel = False
                        a2_ni_sel = (
                            angles[a2][0] - 1 not in sel_indices) and (angles[a2][1] - 1 not in sel_indices) and (angles[a2][2] - 1 not in sel_indices)
                        if a2_ni_sel and is_neighbor(distmat, angles[a1], angles[a2], allow_xx=False, cuttoff=cuttoff):
                            neigh_list.append(a2)
                    if len(neigh_list) > 0:
                        resl.aa_ids_[a1] = {
                            v: aa_idx + k for k, v in enumerate(neigh_list)}
                        aa_idx += len(neigh_list)
                if at and a1_ni_sel:
                    neigh_list = []
                    for t2 in range(n_tor):
                        # a2_ni_sel = False
                        # a2_ni_sel = (angles[a2][0]-1 not in sel_indices) and (angles[a2][1]-1 not in sel_indices)
                        if is_neighbor(distmat, angles[a1], torsions[t2], allow_xx=False, cuttoff=cuttoff):
                            neigh_list.append(t2)
                    if len(neigh_list) > 0:
                        resl.at_ids_[a1] = {
                            v: at_idx + k for k, v in enumerate(neigh_list)}
                        at_idx += len(neigh_list)

        if tt:
            for t1 in range(n_tor):
                neigh_list = []
                for t2 in range(t1 + 1, n_tor):
                    if is_neighbor(distmat, torsions[t1], torsions[t2], allow_xx=False, cuttoff=cuttoff):
                        neigh_list.append(t2)
                if len(neigh_list) > 0:
                    resl.tt_ids_[t1] = {
                        v: tt_idx + k for k, v in enumerate(neigh_list)}
                    at_idx += len(neigh_list)

        return resl


def main():
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument(
        "-p", "--pdb", help="file: pdb format, of molecular system", required=True)
    requiredNamed.add_argument(
        "-t", "--tree", help="tree-file of molecular system", required=True)
    parser.add_argument(
        "-s", "--sel", help="sel_fliter string for atoms involving which bonds and angles are not-considered for entropy calculation: as for cpptraj", default="@H=")
    parser.add_argument("-c", "--cutoff", help="distance cutoff for neighbor consideration",
                        type=int, action="store", default=[6], nargs='*')
    parser.add_argument("-osf", "--out-subset-file",
                        help="filename for subset output", default="subset.txt")
    parser.add_argument(
        "-d", "--dist", help="atom-atom distance file molecular system")
    parser.add_argument("-onfp", "--out-neighbor-file-prefix",
                        help="filename prefix for neighbor output", default="neigh")
    parser.add_argument("-bo", "--bond-order",
                        help="bond-order for entropy calculation", type=int, default=2)
    parser.add_argument("-ao", "--angle-order",
                        help="angle-order for entropy calculation", type=int, default=2)
    parser.add_argument("-to", "--torsion-order",
                        help="torsion-order for entropy calculation", type=int, default=2)
    parser.add_argument("-nba", "--no-ba-cross",
                        help="include bond-angle-cross in entropy calculation", action="store_false")
    parser.add_argument("-nbt", "--no-bt-cross",
                        help="include bond-torsion-cross in entropy calculation", action="store_false")
    parser.add_argument("-nat", "--no-at-cross",
                        help="include bond-torsion-cross in entropy calculation", action="store_false")
    args = parser.parse_args()

    print("Reading: Mol. System Tree..")
    tree = mmsystem.MMTree()
    tree.load(args.tree, header=True, sel_filter=args.sel)
    moltree = tree.to_dataframe()
    # moltree = read_tree_as_dataframe(args.tree, args.pdb, args.sel)
    print("Getting: indices of selected atoms for subset determination...")
    pdb = pt.load(args.pdb)
    selindices = pdb.top.select(args.sel)
#
#     print("Writing: subset input file for entropy calculation...")
#     n_bnd_eff, n_ang_eff, n_tor_eff = write_subsetfile(
#         moltree, args.out_subset_file)
#     print("eff-bonds: {}, eff-angles: {}, torsions: {}".format(n_bnd_eff,
#                                                                n_ang_eff, n_tor_eff))
#     if args.dist:
#         print("Loading: Atom-atom distance matrix...")
#         distances = np.loadtxt(args.dist)
#
#         for cu in args.cutoff:
#             print("Building: neighbor-list input file with distance cutoff(" +
#                   str(cu) + ") for entropy calc...")
#             neighbors = neighbor_builder(distances, moltree, selindices,
#                                          args.bond_order == 2, args.angle_order == 2, args.torsion_order == 2,
#                                          args.no_ba_cross, args.no_bt_cross, args.no_at_cross, cuttoff=cu)
#             print("Writing: neighbor-list input file with distance cutoff(" +
#                   str(cu) + ") for entropy calc...")
#             write_neighbor(neighbors, n_bnd_eff, n_ang_eff, n_tor_eff,
# args.out_neighbor_file_prefix + "_" + str(cu) + ".txt")


# if __name__ == "__main__":
#     main()
