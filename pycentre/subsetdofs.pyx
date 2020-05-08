import numpy as np
import pytraj as pt
import pandas as pd
import copy
import argparse

try:
    from pycentre import utils_ as utls
    from pycentre import mmsystem
    from pycentre.datatypes import *
except ImportError:
    from . import utils_ as utls
    from . import mmsystem
    from .datatypes import *


class SubsetDOFs:
    def __init__(self,  sel_filter="@H=", filter_dofs=['bond', 'angle'], in_sel_all=False, n_bnd=None, n_ang=None, n_tor=None, entropy_workset=BATSet.XY2D):
        cdef int k, v
        if isinstance(entropy_workset, BATSet):
            self.sel_filter = sel_filter
            self.filters = filter_dofs
            self.in_sel_all = in_sel_all
            self.moltree = None

            self.bnd_ids_ = {}
            self.bnd_idx_ = {}
            self.ang_ids_ = {}
            self.ang_idx_ = {}
            self.tor_ids_ = {}
            self.tor_idx_ = {}

            self.n_bnd_tot = 0
            self.n_ang_tot = 0
            self.n_tor_tot = 0

            self.n_bnd_eff = self.n_bnd_tot
            self.n_ang_eff = self.n_ang_tot
            self.n_tor_eff = self.n_tor_tot

            if n_bnd is not None and n_ang is not None and n_tor is not None:
                if entropy_workset.has(BATSet.B1D):
                    self.n_bnd_tot = n_bnd
                    for k, v in enumerate(range(self.n_bnd_tot)):
                        self.bnd_ids_[v] = k
                        self.bnd_idx_[k] = v

                if entropy_workset.has(BATSet.A1D):
                    self.n_ang_tot = n_ang
                    for k, v in enumerate(range(self.n_bnd_tot)):
                        self.ang_ids_[v] = k
                        self.ang_idx_[k] = v

                if entropy_workset.has(BATSet.D1D):
                    self.n_tor_tot = n_tor
                    for k, v in enumerate(range(self.n_bnd_tot)):
                        self.tor_ids_[v] = k
                        self.tor_idx_[k] = v
        else:
            raise Exception("ERROR: SubsrtDOFs.__init__(entropy_workset must be an instanceof datatypes.BATSet)")

    def equals(self, other, matchtree=True):
        ismatchtree = True
        if matchtree:
            ismatchtree = self.moltree is not None and other.moltree is not None
            ismatchtree = ismatchtree and self.moltree.equals(other.moltree)
        if ismatchtree:
            isb = self.n_bnd_tot == other.n_bnd_tot and self.n_bnd_eff == other.n_bnd_eff
            isa = isb and self.n_ang_tot == other.n_ang_tot and self.n_ang_eff == other.n_ang_eff
            ist = isa and self.n_tor_tot == other.n_tor_tot and self.n_tor_eff == other.n_tor_eff
            isbl = ist and sorted(self.bnd_ids_) == sorted(other.bnd_ids_)
            isal = isbl and sorted(self.ang_ids_) == sorted(other.ang_ids_)
            istl = isal and sorted(self.tor_ids_) == sorted(other.tor_ids_)
            return istl
        else:
            return False

    def set_moltree(self, moltree):
        self.moltree = moltree
        if isinstance(moltree, mmsystem.MMTree):
            self.moltree = moltree.to_dataframe()

        self.n_bnd_tot = np.max(self.moltree['bnd_idx'])
        self.n_ang_tot = np.max(self.moltree['ang_idx'])
        self.n_tor_tot = np.max(self.moltree['tor_idx'])

        self.n_bnd_eff = self.n_bnd_tot
        self.n_ang_eff = self.n_ang_tot
        self.n_tor_eff = self.n_tor_tot

        self.bnd_ids_ = {}
        self.bnd_idx_ = {}
        self.ang_ids_ = {}
        self.ang_idx_ = {}
        self.tor_ids_ = {}
        self.tor_idx_ = {}

        cdef int bnd_all=2, ang_all=3, tor_all = 4
        cdef int k, v
        if not self.in_sel_all:
            bnd_all, ang_all, tor_all = 0, 0, 0

        # Negative resids are  used for pseudos, first -1, second pseudo -2 an so on
        if 'bond' in self.filters:
            bnd_ids = np.array(self.moltree[(self.moltree['bflt'] == bnd_all) & (
                self.moltree['bnd_idx'] > 0) | (self.moltree['bnd_res'] < 0)]['bnd_idx'])
            self.n_bnd_eff = len(bnd_ids)
            for k, v in enumerate(bnd_ids):
                self.bnd_ids_[v-1] = k
                self.bnd_idx_[k] = v-1
        else:
            for k, v in enumerate(range(self.n_bnd_tot)):
                self.bnd_ids_[v-1] = k
                self.bnd_idx_[k] = v-1

        if 'angle' in self.filters:
            ang_ids = np.array(self.moltree[(self.moltree['aflt'] == ang_all) & (
                self.moltree['ang_idx'] > 0) | (self.moltree['ang_res'] < 0)]['ang_idx'])
            self.n_ang_eff = len(ang_ids)
            for k, v in enumerate(ang_ids):
                self.ang_ids_[v-1] = k
                self.ang_idx_[k] = v-1
        else:
            for k, v in enumerate(range(self.n_ang_tot)):
                self.ang_ids_[v-1] = k
                self.ang_idx_[k] = v-1

        if 'torsion' in self.filters:
            tor_ids = np.array(self.moltree[(self.moltree['tflt'] == tor_all) & (
                self.moltree['tor_idx'] > 0) | (self.moltree['tor_res'] < 0)]['tor_idx'])

            self.n_tor_eff = len(tor_ids)
            for k, v in enumerate(tor_ids):
                self.tor_ids_[v-1] = k
                self.tor_idx_[k] = v-1
        else:
            for k, v in enumerate(range(self.n_tor_tot)):
                self.tor_ids_[v] = k
                self.tor_idx_[k] = v

    def get_pseudo_ids(self, what):
        # Negative resids are  used for pseudos, first -1, second pseudo -2 an so on
        if what.lower() in ['bond', 'b']:
            return np.array(self.moltree[(self.moltree['bnd_res'] < 0)]['bnd_idx'])
        elif what.lower() in ['angle', 'a']:
            return np.array(self.moltree[(self.moltree['ang_res'] < 0)]['ang_idx'])
        elif what.lower() in ['torsion', 't']:
            return np.array(self.moltree[(self.moltree['tor_res'] < 0)]['tor_idx'])
        else:
            print(
                "`what` can be one of ['bond', 'b', 'angle', 'a', 'torsion', 't'], supplied: " + str(what))
            return None

    def get_pseudo_indices(self, what):
        # Negative resids are  used for pseudos, first -1, second pseudo -2 an so on
        if what.lower() in ['bond', 'b']:
            return [self.get_id2index(what, i) for i in np.array(self.moltree[(self.moltree['bnd_res'] < 0)]['bnd_idx'])]
        elif what.lower() in ['angle', 'a']:
            return [self.get_id2index(what, i) for i in np.array(self.moltree[(self.moltree['ang_res'] < 0)]['ang_idx'])]
        elif what.lower() in ['torsion', 't']:
            return [self.get_id2index(what, i) for i in np.array(self.moltree[(self.moltree['tor_res'] < 0)]['tor_idx'])]
        else:
            print(
                "`what` can be one of ['bond', 'b', 'angle', 'a', 'torsion', 't'], supplied: " + str(what))
            return None

    def get_id2index(self, what, id_):
        if what.lower() in ['bond', 'b']:
            return self.bnd_ids_.get(id_)
        elif what.lower() in ['angle', 'a']:
            return self.ang_ids_.get(id_)
        elif what.lower() in ['torsion', 't']:
            return self.tor_ids_.get(id_)
        else:
            print(
                "`what` can be one of ['bond', 'b', 'angle', 'a', 'torsion', 't'], supplied: " + str(what))
            return None

    def get_index2id(self, what, idx_):
        if what.lower() in ['bond', 'b']:
            return self.bnd_idx_.get(idx_)
        elif what.lower() in ['angle', 'a']:
            return self.ang_idx_.get(idx_)
        elif what.lower() in ['torsion', 't']:
            return self.tor_idx_.get(idx_)
        else:
            print(
                "`what` can be one of ['bond', 'b', 'angle', 'a', 'torsion', 't'], supplied: " + str(what))

    def write(self, subsetoutfile="subset.txt"):
        with open(subsetoutfile, "w") as fOut:
            if self.bnd_ids_:
                fOut.write("bond " + str(self.n_bnd_tot) + " ")
                fOut.write(str(len(self.bnd_ids_)))
                for b in sorted(self.bnd_ids_.keys()):
                    fOut.write(" " + str(b))
                fOut.write(";\n")

            if self.ang_ids_:
                fOut.write("angle " + str(self.n_ang_tot) + " ")
                fOut.write(str(len(self.ang_ids_)))
                for a in sorted(self.ang_ids_.keys()):
                    fOut.write(" " + str(a))
                fOut.write(";\n")

            if self.tor_ids_:
                fOut.write("torsion " + str(self.n_tor_tot) + " ")
                fOut.write(str(len(self.tor_ids_)))
                for t in sorted(self.tor_ids_.keys()):
                    fOut.write(" " + str(t))
                fOut.write(";\n")

    def load(self, subsetfile, entropy_workset=BATSet.XY2D):
        if isinstance(entropy_workset, BATSet):
            with open(subsetfile, "r") as fin:
                for ln in utls.readlines_delim(fin, ";"):
                    words = ln.split()
                    if len(words) > 3:
                        dtype = words[0]
                        n_dim_tot = int(words[1])
                        n_dim_eff = int(words[2])
                        #dims = sorted(dims)
                        if dtype == 'bond' and entropy_workset.has(BATSet.B1D):
                            dims = map(int, words[3:])
                            self.n_bnd_tot = n_dim_tot
                            self.n_bnd_eff = n_dim_eff
                            for k, v in enumerate(dims):
                                self.bnd_ids_[v] = k
                                self.bnd_idx_[k] = v
                        if dtype == 'angle' and entropy_workset.has(BATSet.A1D):
                            dims = map(int, words[3:])
                            self.n_ang_tot = n_dim_tot
                            self.n_ang_eff = n_dim_eff
                            for k, v in enumerate(dims):
                                self.ang_ids_[v] = k
                                self.ang_idx_[k] = v
                        if dtype == 'torsion' and entropy_workset.has(BATSet.D1D):
                            dims = map(int, words[3:])
                            self.n_tor_tot = n_dim_tot
                            self.n_tor_eff = n_dim_eff
                            for k, v in enumerate(dims):
                                self.tor_ids_[v] = k
                                self.tor_idx_[k] = v
        else:
            raise Exception("ERROR: SubsrtDOFs.load(entropy_workset must be an instanceof datatypes.BATSet)")

