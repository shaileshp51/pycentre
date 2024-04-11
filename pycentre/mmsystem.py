# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pytraj as pt
import pandas

from pkg_resources import resource_filename
libfilename = resource_filename('pycentre', 'data/sidechaintorsions.lib')

from abc import ABCMeta, abstractmethod
from re import compile as re_compile
from re import match as re_match


class Residue2AminoAcidMap:
    def __init__(self):
        self.residue2aa = {
            'ALA': 'ALA',
            'ARG': 'ARG',
            'ASP': 'ASP',
            'ASN': 'ASN',

            'CYS': 'CYS',
            'CYX': 'CYS',

            'GLU': 'GLU',
            'GLN': 'GLN',
            'GLY': 'GLY',

            'HIS': 'HIS',
            'HID': 'HIS',
            'HIE': 'HIS',
            'HIP': 'HIS',
            'HSD': 'HSD',
            'HSE': 'HSE',
            'HSP': 'HSP',

            'ILE': 'ILE',

            'LEU': 'LEU',
            'LYS': 'LYS',

            'MET': 'MET',

            'PRO': 'PRO',
            'PHE': 'PHE',

            'SER': 'SER',

            'THR': 'THR',
            'TRP': 'TRP',
            'TYR': 'TYR',

            'VAL': 'VAL'
        }

    def res2aa(self, resname):
        return self.residue2aa.get(resname, False)


class TorisonNamessLib:
    def __init__(self, libfile):
        import pandas as pd
        self.df = pd.read_csv(libfile, index_col=None)
        self.res2aa_map = Residue2AminoAcidMap()

    def sidechain_torname(self, resname, aname1, aname2, aname3, aname4):
        torname = 'UNNAMED'
        astr14 = '%s-%s-%s-%s' % (aname1, aname2, aname3, aname4)
        aaname = self.res2aa_map.res2aa(resname)
        if aaname:
            s = self.df.loc[(self.df['SIDE_CHAIN'] == aaname) & (
                self.df['ATOMS_IN_DEFINITION'] == astr14), 'NAME']
            if not s.empty:
                torname = s.values[0]
            else:
                astr14 = '%s-%s-%s-%s' % (aname4, aname3, aname2, aname1)
                s1 = self.df.loc[(self.df['SIDE_CHAIN'] == aaname) & (
                    self.df['ATOMS_IN_DEFINITION'] == astr14), 'NAME']
                if not s1.empty:
                    torname = s1.values[0]
        return torname

    def torsion_name(self, res1, res2, res3, res4, rname1, rname2, rname3, rname4, aname1, aname2, aname3, aname4):
        torname = 'UNNAMED'
        if res1 == res2 and res1 == res3 and res1 == res4 and rname1 == rname2 and rname1 == rname3 and rname1 == rname4:
            if aname1 == "N" and aname2 == "CA" and aname3 == "C" and aname4 == "O":
                torname = 'O-PSI'
                return torname
            elif aname1 == "O" and aname2 == "C" and aname3 == "CA" and aname4 == "N":
                torname = 'O-PSI'
                return torname
            else:
                torname = self.sidechain_torname(
                    rname1, aname1, aname2, aname3, aname4)
        elif res1 == res2 and res3 == res4 and res2 == res3 - 1:
            if aname1 == "CA" and aname2 == "N" and aname3 == "C" and aname4 == "CA":
                torname = 'OMEGA'
                return torname
            elif aname1 == "H" and aname2 == "N" and aname3 == "C" and aname4 == "CA":
                torname = 'H-OMEGA'
                return torname
        elif res1 == res2 and res3 == res4 and res3 == res2 - 1:
            if aname4 == "CA" and aname3 == "C" and aname2 == "N" and aname1 == "CA":
                torname = 'OMEGA'
                return torname
            elif aname4 == "CA" and aname3 == "C" and aname2 == "N" and aname1 == "H":
                torname = 'H-OMEGA'
                return torname
        elif res1 == res2 - 1 and res2 == res3 and res2 == res4:
            if aname1 == "C" and aname2 == "N" and aname3 == "CA" and aname4 == "C":
                torname = 'PHI'
                return torname
            elif aname4 == "HA" and aname3 == "CA" and aname2 == "N" and aname1 == "C":
                torname = 'HA-PHI'
                return torname
        elif res1 == res2 and res1 == res3 and res4 == res1 - 1:
            if aname4 == "C" and aname3 == "N" and aname2 == "CA" and aname1 == "C":
                torname = 'PHI'
                return torname
            elif aname4 == "C" and aname3 == "N" and aname2 == "CA" and aname1 == "HA":
                torname = 'HA-PHI'
                return torname

        elif res1 == res2 and res1 == res3 and res1 == res4 - 1:
            if aname1 == "N" and aname2 == "CA" and aname3 == "C" and aname4 == "N":
                torname = 'PSI'
                return torname
        elif res4 == res2 and res4 == res3 and res4 == res1 - 1:
            if aname4 == "N" and aname3 == "CA" and aname2 == "C" and aname1 == "N":
                torname = 'PSI'
        if torname == 'UNNAMED' and (aname1[0] == 'H' or aname4[0] == 'H'):
            torname = 'CHI-H'
        return torname


class MMTreeNode:
    def __init__(self, a1, a2, a3, a4, tor_type, tor_phase):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a1_res = 0
        self.a2_res = 0
        self.a3_res = 0
        self.a4_res = 0
        self.bnd_res = 0
        self.ang_res = 0
        self.tor_res = 0
        self.bnd_idx = 0
        self.ang_idx = 0
        self.tor_idx = 0
        self.bflt = 0
        self.aflt = 0
        self.tflt = 0
        self.tor_type = tor_type
        self.tor_phase = tor_phase
        self.name = 'UNNAMED'
        self.tor_atom_names = ''
        self.is_bb = False

    def __str__(self):
        fmt = """%6i %6i %6i %6i %6i %6i %6i %6i %7i %7i %7i %7i %7i %7i %4i %4i %4i %8s %9i %9s %19s"""
        tmp = fmt % (self.a1, self.a2, self.a3, self.a4,
                     self.a1_res, self.a2_res, self.a3_res, self.a4_res,
                     self.bnd_res, self.ang_res, self.tor_res,
                     self.bnd_idx, self.ang_idx, self.tor_idx,
                     self.bflt, self.aflt, self.tflt,
                     self.tor_type, self.tor_phase, self.name, self.tor_atom_names
                     )
        return tmp

    def str_csv(self):
        return ", ".join([self.a1, self.a2, self.a3, self.a4,
                          self.a1_res, self.a2_res, self.a3_res, self.a4_res,
                          self.bnd_res, self.ang_res, self.tor_res,
                          self.bnd_idx, self.ang_idx, self.tor_idx,
                          self.bflt, self.aflt, self.tflt,
                          self.tor_type, self.tor_phase, self.name, self.tor_atom_names, self.is_bb]
                         )


class MMTree:
    def __init__(self, topofile=None, sel_build=None, sel_filter=None):
        self.nodes = {}
        self.topofile = topofile
        self.sel_build = sel_build
        self.sel_filter = sel_build if sel_filter is None else sel_filter

    def add_node(self, node_id, node):
        if isinstance(node, MMTreeNode):
            if node.a1 == node_id:
                self.nodes[node_id] = node
            else:
                raise Exception(
                    "node_id(%d) does not match node.a1(%d)\n" % (node_id, node.a1))
        else:
            raise Exception("node is not of type(TreeNode)\n")

    def __str__(self):
        fmt = """%6s %6s %6s %6s %6s %6s %6s %6s %7s %7s %7s %7s %7s %7s %4s %4s %4s %8s %9s %9s %19s"""
        head = fmt % ('a1', 'a2', 'a3', 'a4', 'a1_res', 'a2_res', 'a3_res', 'a4_res',
                      'bnd_res', 'ang_res', 'tor_res', 'bnd_idx', 'ang_idx', 'tor_idx',
                      'bflt', 'aflt', 'tflt', 'tor_type', 'tor_phase', 'tor_name', 'tor_atom_names'
                      )

        s1 = '{}={}'.format('__topology__', self.topofile)
        s2 = '{}={}'.format('__sel_build__', self.sel_build)
        s3 = '{}={}'.format('__sel_filter__', self.sel_filter)
        lst = [s1, s2, s3, head]
        self._assign_filter(self.sel_filter)
        for k in sorted(self.nodes.keys()):
            lst.append(str(self.nodes[k]))
        return '\n'.join(lst)

    def str_csv(self):
        head = ", ".join(['a1', 'a2', 'a3', 'a4', 'a1_res', 'a2_res', 'a3_res', 'a4_res',
                          'bnd_res', 'ang_res', 'tor_res', 'bnd_idx', 'ang_idx', 'tor_idx',
                          'bflt', 'aflt', 'tflt', 'tor_type', 'tor_phase', 'tor_name', 'tor_atom_names'
                          ])
        lst = [head]
        self._assign_filter(self.sel_filter)
        for k in sorted(self.nodes.keys()):
            lst.append(self.nodes[k].str_csv())
        return '\n'.join(lst)

    def save(self, filename):
        with open(filename, 'w') as fs:
            fs.write('{}={}\n'.format('__topology__', self.topofile))
            fs.write('{}={}\n'.format('__sel_build__', self.sel_build))
            fs.write('{}={}\n'.format('__sel_filter__', self.sel_filter))
            self._assign_filter(self.sel_filter)
            fs.write(str(self))

    def _assign_filter(self, sel_filter=None):
        if sel_filter is not None:
            print("Assigning sel_filter membership(bflt,aflt,tflt): '" +
                  self.topofile + "'")
            data_len = len(self.nodes)
            pdb = pt.load_topology(self.topofile)
            H_indices = pdb.select(sel_filter)
            for i in range(1, data_len + 1):
                # Get selected atoms # in bond
                if self.nodes[i].a2 != -1:
                    self.nodes[i].bflt = 0
                    if self.nodes[i].a2 - 1 in H_indices:
                        self.nodes[i].bflt += 1
                    if self.nodes[i].a1 - 1 in H_indices:
                        self.nodes[i].bflt += 1
                # Get selected atoms # in angle
                if not (self.nodes[i].a2 == -1 or self.nodes[i].a3 == -1):
                    self.nodes[i].aflt = 0
                    if self.nodes[i].a3 - 1 in H_indices:
                        self.nodes[i].aflt += 1
                    if self.nodes[i].a1 - 1 in H_indices:
                        self.nodes[i].aflt += 1
                # Get selected atoms # in torsion
                if not (self.nodes[i].a2 == -1 or self.nodes[i].a3 == -1 or self.nodes[i].a4 == -1):
                    self.nodes[i].tflt = 0
                    if self.nodes[i].a1 - 1 in H_indices:
                        self.nodes[i].tflt += 1
                    if self.nodes[i].a2 - 1 in H_indices:
                        self.nodes[i].tflt += 1
                    if self.nodes[i].a3 - 1 in H_indices:
                        self.nodes[i].tflt += 1
                    if self.nodes[i].a4 - 1 in H_indices:
                        self.nodes[i].tflt += 1

    def _to_node(self, node_dict):
        obj = None
        try:
            obj = MMTreeNode(node_dict['a1'], node_dict['a2'], node_dict['a3'],
                             node_dict['a4'], node_dict['tor_type'], node_dict['tor_phase'])
            obj.a1_res = node_dict['a1_res']
            obj.a2_res = node_dict['a2_res']
            obj.a3_res = node_dict['a3_res']
            obj.a4_res = node_dict['a4_res']
            obj.bnd_res = node_dict['bnd_res']
            obj.ang_res = node_dict['ang_res']
            obj.tor_res = node_dict['tor_res']
            obj.bnd_idx = node_dict['bnd_idx']
            obj.ang_idx = node_dict['ang_idx']
            obj.tor_idx = node_dict['tor_idx']
            obj.bflt = node_dict['bflt']
            obj.aflt = node_dict['aflt']
            obj.tflt = node_dict['tflt']
            obj.name = node_dict['tor_name']
            obj.tor_atom_names = node_dict['tor_atom_names']
            obj.is_bb = False
        except Exception:
            raise Exception("MMTreeNode parsing error")
        return obj

    def load(self, filename, header=True, sel_filter=None):
        """
            returns a pandas dataframe, where columns from BAT residue memberships
            number of hydrogens involved in bond/angle/torsion (columns: xxx_sel) of df
            index of bond/angle/torsion (1-based) (columns: xxx_idx) of df

            filename: filename from which tree will be load
            pdbfile: filename of corresponding pdb
            header: True::load tree heads from file, False::give heads as V??

        """
        head_to_read = True
        with open(filename, "r") as f:
            heads = []
            node_id = 0
            self.nodes = {}
            for ri, row in enumerate(f.readlines()):
                row_txt = row.strip()
                if len(row_txt) > 0 and row_txt[0] == '#':
                    pass
                elif len(row_txt) > 1 and row_txt[0:2] == '__':
                    eqidx = row_txt.find('=')
                    words = (row_txt[:eqidx], row_txt[eqidx + 1:])
                    if len(words) > 1:
                        if words[0] == '__topology__':
                            self.topofile = words[1]
                        elif words[0] == '__sel_build__':
                            self.sel_build = words[1]
                        elif words[0] == '__sel_filter__':
                            self.sel_filter = words[1]
                        else:
                            raise Exception("Treefile header is corrupt")
                elif head_to_read:
                    for k in row.split():
                        heads.append(k.strip())
                    head_to_read = False
                else:
                    row_dict = {}
                    for i, c in enumerate(row.split()):
                        if heads[i] in ['tor_type', 'tor_name', 'tor_atom_names']:
                            row_dict[heads[i]] = c.strip()
                        elif heads[i] in ['is_bb']:
                            row_dict[heads[i]] = c.strip() == 'True'
                        else:
                            row_dict[heads[i]] = int(c.strip())
                    obj = self._to_node(row_dict)
                    if obj is not None:
                        node_id += 1
                        self.add_node(node_id, obj)
        if self.sel_filter != sel_filter:
            self._assign_filter(sel_filter)

    def to_dataframe(self):
        result_dlist = [
            self.nodes[k].__dict__ for k in sorted(self.nodes.keys())]
        result = pandas.DataFrame(result_dlist)
        return result


class MMSystem:
    # __metaclass__ = ABCMeta
    """
        This is a class for representing a Molecular Dynamics System
        It reads the topology of the a protein/small molecule or a receptor-ligand
        system and constructs a moecule-tree required for converting Cartesian Coordinate
        Trajectory of the system to an Internal Coordinate System (Bond/Angle/Torsion)
        Trajectory.
        It also can represent torsions of a rigid group containing two or three common atoms as 
        a main-torsion and other as phase angles.
        (e.g for a methyle group it can have one main-tosion and three phase-torsions) 
    """

    def __init__(self, sel_build=':*@*', sel_phaseref=':*@N,CA,C', sel_filter=None, atom_start_index=1):
        self.sel_build = sel_build
        self.sel_phaseref = sel_phaseref
        self.sel_filter = self.sel_build if sel_filter is None else sel_filter
        self.atom_start_index = atom_start_index
        self.top = None
        self.topopfile = ''
        self.atoms = {}
        self.residues = {}
        self.resioff = {}
        self.bonds = []
        self.phaserefs = {}
        self.tree = None
        self.tornamelib = TorisonNamessLib(libfilename)

    def load_topology(self, topofile):
        topo = pt.load_topology(topofile)
        self.top = topo
        self.atomselected = self.top.select(self.sel_build)
        phaserefsatoms = self.top.select(self.sel_phaseref)
        self.atoms = {
            a.index + self.atom_start_index: a for a in self.top.atoms if a.index in self.atomselected}
        self.phaserefs = {a.index + 1: False for a in self.top.atoms}
        for a in self.top.atoms:
            if a.index in phaserefsatoms:
                self.phaserefs[a.index + 1] = True
        atmlist = [a for a in self.top.atoms]
        reslist = [a for a in self.top.residues]
        self.resioff = {atmlist[r.first].resid: r.original_resid - atmlist[r.first].resid for r in self.top.residues}
        self.residues = {
            r.original_resid: r for r in self.top.residues}
        self.bonds = [tuple([i + self.atom_start_index for i in b.indices])
                      for b in self.top.bonds if b.indices[0] in self.atomselected and b.indices[1] in self.atomselected]
        self.topofile = topofile
        self.tree = MMTree(topofile=self.topofile,
                           sel_build=self.sel_build, sel_filter=self.sel_filter)

    def num_atoms(self):
        return len(self.atoms)

    def atom_indices(self):
        return [a - 1 for a in sorted(self.atoms.keys())]

    def create_bat_tree(self, useDFS, roots, usePhase, pseudo_bonds):
        # pseudo will be mutated using pop(-1), so make a copy of pseudo_bonds
        pseudo = list(pseudo_bonds)
        # isBackboneAtom = {}
        connect = {}
        self.tree = MMTree(self.topofile, self.sel_build, self.sel_filter)
#         for k, a in self.atoms.items():
#             if (a.name in bbAtomNames):
#                 isBackboneAtom[k] = True
#             else:
#                 isBackboneAtom[k] = False

        logger.debug("Number of bonds: %i" % len(self.bonds))
        logger.debug("Bonds: %s" % str(self.bonds))
        for b in self.bonds:
            if (b[0] in self.atoms) and (b[1] in self.atoms):
                if (b[0] in connect):
                    (connect[b[0]]).append(b[1])
                else:
                    connect[b[0]] = [b[1]]
                if (b[1] in connect):
                    (connect[b[1]]).append(b[0])
                else:
                    connect[b[1]] = [b[0]]
            elif((b[0] not in self.atoms) and (b[1] not in self.atoms)):
                logger.debug("Probably solvent...Atom No(" + b[0] + ")")
            else:
                msg = "Error: residue selection is not a complete molecule, there is a bond between atoms " + \
                    str(b[0]) + " and " + str(b[1]) + "\n"
                logger.error(msg)
                raise Exception(msg)

        for ki in connect.keys():
            connect[ki].sort()
        logger.info("Building connectivity complete...")
        logger.debug(connect)
        logger.debug(roots)
        self.tree.add_node(roots[0], MMTreeNode(roots[0], -1, -1, -1, 'n', -1))
        self.tree.add_node(roots[1], MMTreeNode(
            roots[1], roots[0], -1, -1, 'n', -1))
        self.tree.add_node(roots[2], MMTreeNode(
            roots[2], roots[1], roots[0], -1, 'n', -1))

        nodes = []

        # Check if root1 is terminal
        for tmp in connect[roots[0]]:
            if (tmp not in self.tree.nodes):
                msg = "Error building tree: Root atom1 is connected to non-root atom" + \
                    str(tmp)
                logger.error(msg)
                raise Exception(msg)

        # Deal with root 2 impropers if necessary
        n_non_terminal_links_with_root2 = 0
        id_non_terminal_links_with_root2 = -1
        tmp2s = []
        for tmp in connect[roots[1]]:
            if (tmp not in self.tree.nodes):
                for tmp2 in connect[tmp]:
                    if (tmp2 not in self.tree.nodes):
                        n_non_terminal_links_with_root2 += 1
                        id_non_terminal_links_with_root2 = tmp2
                        tmp2s.append(str(tmp2))
                        break
        if (n_non_terminal_links_with_root2 > 1):
            msg = """Error building tree: atom[%s] is
                     connected to non-root atom[%s]\n
                     All but except atmost one atom connected to
                     root-atom-2 must be terminal or root atoms\n""" % (str(tmp), ",".join(tmp2s))
            logger.error(msg)
            raise Exception(msg)
        elif n_non_terminal_links_with_root2 == 1:
            # only one non-terminal and non-root atoms is connected to mid-root
            # atom (root2)
            for tmp in connect[roots[1]]:
                if (tmp not in self.tree.nodes):
                    self.tree.add_node(tmp, MMTreeNode(
                        tmp, roots[1], roots[0], roots[2], 'i', -1))
                    if useDFS:
                        nodes.extend([tmp, roots[2]])
                    else:
                        nodes.extend([roots[2], tmp])
        else:
            # only terminal atoms are connected to mid-root atom (root2)
            for tmp in connect[roots[1]]:
                if (tmp not in self.tree.nodes):
                    self.tree.add_node(tmp, MMTreeNode(
                        tmp, roots[1], roots[0], roots[2], 'i', -1))
                    for tmp2 in connect[tmp]:
                        if (tmp2 not in self.tree.nodes):
                            msg = """Error building tree: Atom $tmp is
                                     connected to non-root atom $tmp2\n
                                     All atoms connected to Root-atom-2 must
                                     be terminal or root atoms\n"""
                            logger.error(msg)
                            raise Exception(msg)
            nodes.append(roots[2])

        # Check that root3 has connections
        i = 0
        for tmp in connect[roots[2]]:
            if (tmp not in self.tree.nodes):
                i += 1
        if (i == 0):
            msg = "Error building tree: Root atom 3 is a terminal atom\n"
            logger.error(msg)
            raise Exception(msg)

        pseudo_count = len(pseudo) // 2
        if pseudo_count * 2 != len(pseudo):
            logger.error("Pseudo bonds, should have pair of atoms\n")
            raise Exception("Pseudo bonds, should have pair of atoms\n")
        pop_item_index = -1 if useDFS else 0
        while (pseudo_count >= 0):
            while (len(nodes) > 0):
                branchHead = nodes.pop(pop_item_index)
                # if branchHead not in self.tree.nodes:
                all_twigs = connect[branchHead]
                twigs = []
                for tmp in all_twigs:
                    if (tmp not in self.tree.nodes):
                        twigs.append(tmp)
                for tmp in twigs:
                    third = self.tree.nodes[branchHead].a2
                    fourth = self.tree.nodes[branchHead].a3
                    self.tree.add_node(tmp, MMTreeNode(
                        tmp, branchHead, third, fourth, 'p', -1))
                nodes.extend(twigs)
            pseudo_count -= 1
            if(pseudo_count >= 0):
                # After pop(-1) curr [-2] will become [-1]
                pseudo1 = pseudo.pop(-1)
                pseudo2 = pseudo.pop(-1)
                nodes.append(pseudo1)
                if (pseudo1 in connect):
                    connect[pseudo1].append(pseudo2)
                else:
                    connect[pseudo1] = [pseudo2]
                if (pseudo2 in connect):
                    connect[pseudo1].append(pseudo2)
                else:
                    connect[pseudo2] = [pseudo1]

        # Reassign pseudo_count, since it changed to 0 by above while loop
        pseudo_count = len(pseudo_bonds) // 2
        # Assign residues for atoms, bonds, angles, torsions of tree nodes
        pseudo_DOF = {}
        for k in sorted(self.tree.nodes.keys()):
            val = self.tree.nodes[k]
            val.a1_res = self.atoms[k].resid + self.resioff[self.atoms[k].resid]
            val.a2_res = self.atoms[val.a2].resid + self.resioff[self.atoms[val.a2].resid] if val.a2 != -1 else -1
            val.a3_res = self.atoms[val.a3].resid + self.resioff[self.atoms[val.a3].resid] if val.a3 != -1 else -1
            val.a4_res = self.atoms[val.a4].resid + self.resioff[self.atoms[val.a4].resid] if val.a4 != -1 else -1

            # Assign residue for bondl
            if val.a1_res <= val.a2_res:
                val.bnd_res = val.a1_res
            else:
                val.bnd_res = val.a2_res if val.a2_res != -1 else 0

            # Assign residue for angles
            if val.a1_res == val.a2_res and val.a1_res == val.a3_res:
                val.ang_res = val.a1_res
            elif val.a1_res == val.a2_res:
                val.ang_res = val.a1_res if val.a3_res != -1 else 0
            elif val.a1_res == val.a3_res:
                val.ang_res = val.a1_res if val.a3_res != -1 else 0
            elif val.a2_res == val.a3_res:
                val.ang_res = val.a2_res if val.a3_res != -1 else 0

            # Assign residue for torsions
            if val.a2_res <= val.a3_res and val.a4_res != -1:
                val.tor_res = val.a2_res if val.a2_res != -1 else 0
            elif val.a4_res != -1:
                val.tor_res = val.a3_res if val.a3_res != -1 else 0
            elif val.a4_res == -1:
                val.tor_res = 0

            # Assign pseudo bond/angle/torsion residue
            # TODO: comparisons should be made unordered, content comparioson
            if pseudo_count > 0:
                for i in range(0, 2 * pseudo_count - 1, 2):
                    # Pseudo bond member
                    if set([k, val.a2]) == set([pseudo_bonds[i], pseudo_bonds[i + 1]]):
                        val.bnd_res = -1 * (int(i / 2) + 1)
                    # Pseudo angle member
                    if set([k, val.a2]) == set([pseudo_bonds[i], pseudo_bonds[i + 1]]) and val.a3 != -1:
                        val.ang_res = -1 * (int(i / 2) + 1)
                    elif set([val.a2, val.a3]) == set([pseudo_bonds[i], pseudo_bonds[i + 1]]):
                        val.ang_res = -1 * (int(i / 2) + 1)

                    # TODO: Check and assign pseudo torsion member
                    if set([val.a2, val.a3]) == set([pseudo_bonds[i], pseudo_bonds[i + 1]]) and val.a4 != -1:
                        # val.tor_res = -1 * (int(i/2) + 1)
                        ps_id = -1 * (int(i / 2) + 1)
                        d_tor = pseudo_DOF.get(ps_id, [])
                        if 'ta23' not in d_tor:
                            val.tor_res = ps_id
                            d_tor.append('ta23')
                            pseudo_DOF[ps_id] = d_tor
                            print("pseudo_tosion(%d) ta23: %s" % (ps_id, val))
                            # print connect[val.a4]
                    elif set([k, val.a2]) == set([pseudo_bonds[i], pseudo_bonds[i + 1]]) and len(connect[val.a4]) > 1:
                        ps_id = -1 * (int(i / 2) + 1)
                        d_tor = pseudo_DOF.get(ps_id, [])
                        if 'ta4' not in d_tor:
                            val.tor_res = ps_id
                            d_tor.append('ta4')
                            pseudo_DOF[ps_id] = d_tor
                            print("pseudo_tosion(%d) ta4 : %s" % (ps_id, val))
                            # print connect[val.a4]
                    elif set([val.a3, val.a4]) == set([pseudo_bonds[i], pseudo_bonds[i + 1]]) and len(connect[k]) > 1:
                        ps_id = -1 * (int(i / 2) + 1)
                        d_tor = pseudo_DOF.get(ps_id, [])
                        if 'ta1' not in d_tor:
                            val.tor_res = ps_id
                            d_tor.append('ta1')
                            pseudo_DOF[ps_id] = d_tor
                            print("pseudo_tosion(%d) ta1 : %s" % (ps_id, val))
        # Assign indices to bonds, angles and torsiona and names to torsions
        bnd_idx_ = 0
        ang_idx_ = 0
        tor_idx_ = 0
        for k in sorted(self.tree.nodes.keys()):
            val = self.tree.nodes[k]
            res1 = val.a1_res = self.atoms[k].resid + self.resioff[self.atoms[k].resid]
            res2 = val.a2_res = self.atoms[val.a2].resid + \
                self.resioff[self.atoms[val.a2].resid] if val.a2 != -1 else -1
            res3 = val.a3_res = self.atoms[val.a3].resid + \
                self.resioff[self.atoms[val.a3].resid] if val.a3 != -1 else -1
            res4 = val.a4_res = self.atoms[val.a4].resid + \
                self.resioff[self.atoms[val.a4].resid] if val.a4 != -1 else -1
            aname1 = self.atoms[k].name
            aname2 = self.atoms[val.a2].name if val.a2 != -1 else '????'
            aname3 = self.atoms[val.a3].name if val.a3 != -1 else '????'
            aname4 = self.atoms[val.a4].name if val.a4 != -1 else '????'
            rname1 = self.residues[res1].name if res1 > 0 else 'XXX'
            rname2 = self.residues[res2].name if res2 > 0 else 'XXX'
            rname3 = self.residues[res3].name if res3 > 0 else 'XXX'
            rname4 = self.residues[res4].name if res4 > 0 else 'XXX'
            if val.a2 != -1:
                bnd_idx_ += 1
                val.bnd_idx = bnd_idx_
            if val.a2 != -1 and val.a3 != -1:
                ang_idx_ += 1
                val.ang_idx = ang_idx_
            if val.a2 != -1 and val.a3 != -1 and val.a4 != -1:
                tor_idx_ += 1
                val.tor_idx = tor_idx_

            val.tor_atom_names = '''{0:.<4}-{1:.<4}-{2:.<4}-{3:.<4}'''.format(
                aname1, aname2, aname3, aname4)
            rvalid = res1 > 0 and res2 > 0 and res3 > 0 and res4 > 0
            rnvalid = rname1 != 'XXX' and rname2 != 'XXX' and rname3 != 'XXX' and rname4 != 'XXX'
            anvalis = (aname1 is not None) and (aname2 is not None) and (
                aname3 is not None) and (aname4 is not None)
            if rvalid and rnvalid and anvalis:
                if val.tor_type == 'i':
                    val.name = 'IMPROPER'
                else:
                    val.name = self.tornamelib.torsion_name(
                        res1, res2, res3, res4, rname1, rname2, rname3, rname4, aname1, aname2, aname3, aname4)
            else:
                val.name = 'None'

        logger.info("Building BAT tree completed successfully")
        return(self.tree)

    def assign_phase(self):
        isBackboneTorsion = {}
        phaseTmp = {}
        strValue = "Bond, Angle, Torsional Tree for residue selection:\n"
        strValue += "Columns show 4 atom numbers, 1 phase angle assignment, type of torsion\n"
        # Phase angles: 1. Define torsions which include only backbone atoms
        # print tree.keys()
        for k in sorted(self.tree.nodes.keys()):
            if ((self.tree.nodes[k].a2 > 0) and (self.tree.nodes[k].a3 > 0) and (self.tree.nodes[k].a4 > 0) and
                self.phaserefs[k] and self.phaserefs[self.tree.nodes[k].a2] and
                    self.phaserefs[self.tree.nodes[k].a3] and self.phaserefs[self.tree.nodes[k].a4]):
                isBackboneTorsion[k] = True
                self.tree.nodes[k].is_bb = True
            else:
                isBackboneTorsion[k] = False
                self.tree.nodes[k].is_bb = False
        # Phase angles: 2. Select preferential reference torsions (using
        # backbone atom definitions)
        for k in sorted(self.tree.nodes.keys()):
            phasekey = str(self.tree.nodes[k].a2) + \
                "," + str(self.tree.nodes[k].a3)
            if ((phasekey not in phaseTmp) and isBackboneTorsion[k]):
                phaseTmp[phasekey] = k
        # Phase angles: 3. Assign all full torsions and phase angles using tree
        # numbering and output.
        for k in sorted(self.tree.nodes.keys()):
            phasekey = str(self.tree.nodes[k].a2) + \
                "," + str(self.tree.nodes[k].a3)
            if (phasekey not in phaseTmp):
                phaseTmp[phasekey] = k
            self.tree.nodes[k].tor_phase = phaseTmp[phasekey]
        # Phase angles: 4. Delete phase angle references for 'n' tree members
        # (where no torsion exists) and output.
        for k in sorted(self.tree.nodes.keys()):
            if (self.tree.nodes[k].tor_type == 'n'):
                self.tree.nodes[k].tor_phase = 0
        # Phase angles: 5. Renumber the full torsions and phase angles
        # sequentially.
        offset = 0
        phaseDef = [0]
        for k in sorted(self.tree.nodes.keys()):
            if (self.tree.nodes[k].a2 > 0 and self.tree.nodes[k].a3 > 0 and self.tree.nodes[k].a4 > 0):
                # Assign phase angle reference (numata)
                phaseDef.append(self.tree.nodes[k].tor_phase - offset)
            else:
                offset += 1
        return (phaseDef)

    def __str__(self):
        return(str(self.tree))


class NAMDSystem(MMSystem):
    pass


class AMBERSystem(MMSystem):
    pass


class CHARMMSystem(MMSystem):
    pass
