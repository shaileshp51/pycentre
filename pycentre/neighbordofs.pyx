"""
.. module:: neighbrdofs
.. moduleauthor:: shailesh K Panday <shaileshp51@gmail.com>
.. lastupdated:: 27/06/2018
"""


import numpy as np
import pytraj as pt
import pandas as pd
import copy
import argparse

try:
    from pycentre import utils_ as utls
    from pycentre import mmsystem
    from pycentre import subsetdofs
    from pycentre.datatypes import *
except ImportError:
    try:
        from . import utils_ as utls
        from . import mmsystem
        from . import subsetdofs as subsetdofs
        from .datatypes import *

    except ImportError as err:
        raise ("Error import: " + str(err))

class NeighborDOFsIds2IndexMap:
    def __init__(self, neighbor=None):
        self.bb_ids_ = {}
        self.aa_ids_ = {}
        self.tt_ids_ = {}
        self.ba_ids_ = {}
        self.bt_ids_ = {}
        self.at_ids_ = {}
        self.is_ready = False
        if neighbor is not None and isinstance(neighbor, subsetdofs.NeighborDOFs):
            self.neighbor = neighbor
        else:
            pass

    def idpair2index(self, what, e1, e2):
        found_index = None
        what_low = what.lower()
        if what_low == b'bb' or  what_low == 'bb':
            found_index = self.bb_ids_.get(e1).get(e2, None)
        elif what_low == b'aa' or  what_low == 'aa':
            found_index = self.aa_ids_.get(e1).get(e2, None)
        elif what_low == b'tt' or  what_low == 'tt':
            found_index = self.tt_ids_.get(e1).get(e2, None)
        elif what_low == b'ba' or  what_low == 'ba':
            found_index = self.ba_ids_.get(e1).get(e2, None)
        elif what_low == b'bt' or  what_low == 'bt':
            found_index = self.bt_ids_.get(e1).get(e2, None)
        elif what_low == b'at' or  what_low == 'at':
            found_index = self.at_ids_.get(e1).get(e2, None)
        else:
            print(
                "`what` can be one of ['bb', 'aa', 'tt', 'ba', 'bt', 'at'], supplied: " + str(what))
        if found_index is None:
            raise Exception("{0} idpair({1}, {2}) does not exists".format(what.lower(), str(e1), str(e2)))

        return found_index


class NeighborDOFs:
    def __init__(self, subset=None, entropy_workset=BATSet.XY2D):
        self.bb_ids_ = {}
        self.aa_ids_ = {}
        self.tt_ids_ = {}
        self.ba_ids_ = {}
        self.bt_ids_ = {}
        self.at_ids_ = {}
        self.n_bb_pairs_ = 0
        self.n_aa_pairs_ = 0
        self.n_tt_pairs_ = 0
        self.n_ba_pairs_ = 0
        self.n_bt_pairs_ = 0
        self.n_at_pairs_ = 0
        self.subset = subset
        self.idpair2indexmap = None
        if subset is not None and isinstance(self.subset, subsetdofs.SubsetDOFs):
            self._build_from_subset(entropy_workset)

    def get_typewise_pairs_counts(self):
        nenpairs = (self.n_bb_pairs_, self.n_aa_pairs_, self.n_tt_pairs_,\
                    self.n_ba_pairs_, self.n_bt_pairs_, self.n_at_pairs_)
        return nenpairs

    def get_total_pairs_counts(self):
        numpairs = self.n_bb_pairs_ + self.n_aa_pairs_ + self.n_tt_pairs_\
                    + self.n_ba_pairs_ + self.n_bt_pairs_ + self.n_at_pairs_
        return numpairs

    def _initialize_id_pairs_dict(self):
        for k1 in sorted(self.bb_ids_.keys()):
            tmp_dict = {}
            for k2 in self.bb_ids_[k1]:
                tmp_dict[k2] = -1 # -1 indicates an uninitialized value
            self.idpair2indexmap.bb_ids_[k1] = tmp_dict

        for k1 in sorted(self.aa_ids_.keys()):
            tmp_dict = {}
            for k2 in self.aa_ids_[k1]:
                tmp_dict[k2] = -1 # -1 indicates an uninitialized value
            self.idpair2indexmap.aa_ids_[k1] = tmp_dict

        for k1 in sorted(self.tt_ids_.keys()):
            tmp_dict = {}
            for k2 in self.tt_ids_[k1]:
                tmp_dict[k2] = -1 # -1 indicates an uninitialized value
            self.idpair2indexmap.tt_ids_[k1] = tmp_dict

        for k1 in sorted(self.ba_ids_.keys()):
            tmp_dict = {}
            for k2 in self.ba_ids_[k1]:
                tmp_dict[k2] = -1 # -1 indicates an uninitialized value
            self.idpair2indexmap.ba_ids_[k1] = tmp_dict

        for k1 in sorted(self.bt_ids_.keys()):
            tmp_dict = {}
            for k2 in self.bt_ids_[k1]:
                tmp_dict[k2] = -1 # -1 indicates an uninitialized value
            self.idpair2indexmap.bt_ids_[k1] = tmp_dict

        for k1 in sorted(self.at_ids_.keys()):
            tmp_dict = {}
            for k2 in self.at_ids_[k1]:
                tmp_dict[k2] = -1 # -1 indicates an uninitialized value
            self.idpair2indexmap.at_ids_[k1] = tmp_dict

    def setIdpair2IndexMap(self, id2idxmap, init_idpairdict=False):
            if id2idxmap is not None and isinstance(id2idxmap, NeighborDOFsIds2IndexMap):
                self.idpair2indexmap = id2idxmap
                if init_idpairdict:
                    self._initialize_id_pairs_dict()

            else:
                raise Exception("`id2idxmap` must be ean instance of NeighborDOFsIds2IndexMap")


    def _build_from_subset(self, entropy_workset):
        if self.subset is not None \
        and isinstance(self.subset, subsetdofs.SubsetDOFs)\
        and isinstance(entropy_workset, BATSet):
            bnd_ids = sorted(self.subset.bnd_ids_.keys())
            ang_ids = sorted(self.subset.ang_ids_.keys())
            tor_ids = sorted(self.subset.tor_ids_.keys())
            if self.subset.n_bnd_eff > 0 and entropy_workset.has(BATSet.BB2D):
                self.n_bb_pairs_ = 0
                for i, dim_id in enumerate(bnd_ids[:-1]):
                    self.bb_ids_[dim_id] = np.array(sorted(bnd_ids[i + 1:]))
                    self.n_bb_pairs_ += len(bnd_ids) - i - 1

            if self.subset.n_ang_eff > 0 and entropy_workset.has(BATSet.AA2D):
                self.n_aa_pairs_ = 0
                for i, dim_id in enumerate(ang_ids[:-1]):
                    self.aa_ids_[dim_id] = np.array(sorted(ang_ids[i + 1:]))
                    self.n_aa_pairs_ += len(ang_ids) - i - 1

            if self.subset.n_tor_eff > 0 and entropy_workset.has(BATSet.DD2D):
                self.n_tt_pairs_ = 0
                for i, dim_id in enumerate(tor_ids[:-1]):
                    self.tt_ids_[dim_id] = np.array(sorted(tor_ids[i + 1:]))
                    self.n_tt_pairs_ += len(tor_ids) - i - 1

            if self.subset.n_bnd_eff > 0 and self.subset.n_ang_eff > 0 \
            and entropy_workset.has(BATSet.BA2D):
                self.n_ba_pairs_ = 0
                for i, dim_id in enumerate(bnd_ids):
                    self.ba_ids_[dim_id] = np.array(sorted(ang_ids))
                    self.n_ba_pairs_  += len(ang_ids)

            if self.subset.n_bnd_eff > 0 and self.subset.n_tor_eff > 0 \
            and entropy_workset.has(BATSet.BD2D):
                self.n_bt_pairs_ = 0
                for i, dim_id in enumerate(bnd_ids):
                    self.bt_ids_[dim_id] = np.array(sorted(tor_ids))
                    self.n_bt_pairs_  += len(tor_ids)

            if self.subset.n_ang_eff > 0 and self.subset.n_tor_eff > 0 \
            and entropy_workset.has(BATSet.AD2D):
                self.n_at_pairs_ = 0
                for i, dim_id in enumerate(ang_ids):
                    self.at_ids_[dim_id] = np.array(sorted(tor_ids))
                    self.n_at_pairs_ += len(tor_ids)

    def clone(self):
        return copy.deepcopy(self)

    def equals(self, other, matchsubset=True):
        if isinstance(other, NeighborDOFs):
            issubsetmatch = True
            if matchsubset:
                issubsetmatch = self.subset is not None and other.subset is not None
                issubsetmatch = issubsetmatch and self.subset.equals(other.subset)
            if issubsetmatch:
                isbb = self.bb_ids_ == other.bb_ids_
                isaa = isbb and self.aa_ids_ == other.aa_ids_
                istt = isaa and self.tt_ids_ == other.tt_ids_
                isba = istt and self.ba_ids_ == other.ba_ids_
                isbt = isba and self.bt_ids_ == other.bt_ids_
                isat = isbt and self.at_ids_ == other.at_ids_
                return isat
            else:
                print("""INFORMATION:: SubsetDOFs for both objects must be set before comparison.
                Please set SubsetDOFs and try again""")
                return False
        else:
            print("WARNING:: Incompatible objects comparison attemped.")
            return False

    def set_subset(self, subset):
        if isinstance(subset, subsetdofs.SubsetDOFs):
            self.subset = subset
            # self.subset.set_moltree(subset.moltree)
        else:
            raise Exception("Supplied argument must be an instance of SubsetDOFs")

    def get_pseudo_ids(self, what, bnd_psids=None, ang_psids=None, tor_psids=None):
        if what.lower() == 'bb':
            if self.bb_ids_:
                ps_tups = []
                lin_id = 0
                for k in sorted(self.bb_ids_.keys()):
                    if k != bnd_psids:
                        if bnd_psids in self.bb_ids_[k]:
                            elm_idx = self.bb_ids_[k].index(bnd_psids)
                            ps_tups.append((k, bnd_psids, lin_id + elm_idx))
                        elif k == bnd_psids:
                            for eidx, val in enumerate(self.bb_ids_[k]):
                                ps_tups.append((bnd_psids, val, lin_id + eidx))
                        lin_id += len(self.bb_ids_[k])
                return ps_tups
        elif what.lower() == 'ba':
            if self.bb_ids_:
                ps_tups = []
                lin_id = 0
                for k in sorted(self.bb_ids_.keys()):
                    if k != bnd_psids:
                        if bnd_psids in self.bb_ids_[k]:
                            elm_idx = self.bb_ids_[k].index(bnd_psids)
                            ps_tups.append((k, bnd_psids, lin_id + elm_idx))
                        elif k == bnd_psids:
                            for eidx, val in enumerate(self.bb_ids_[k]):
                                ps_tups.append((bnd_psids, val, lin_id + eidx))
                        lin_id += len(self.bb_ids_[k])
                return ps_tups
        elif what.lower() == 'bt':
            pass
        elif what.lower() == 'aa':
            if self.aa_ids_:
                ps_tups = []
                lin_id = 0
                for k in sorted(self.aa_ids_.keys()):
                    if k not in ang_psids:
                        for dim_ps in ang_psids:
                            if dim_ps in self.aa_ids_[k]:
                                elm_idx = self.aa_ids_[k].index(dim_ps)
                                ps_tups.append((k, dim_ps, lin_id + elm_idx))
                    elif k in ang_psids:
                        for eidx, val in enumerate(self.aa_ids_[k]):
                            ps_tups.append((k, val, lin_id + eidx))
                    lin_id += len(self.aa_ids_[k])
                return ps_tups
        elif what.lower() == 'at':
            pass
        elif what.lower() == 'tt':
            if self.tt_ids_:
                ps_tups = []
                lin_id = 0
                for k in sorted(self.tt_ids_.keys()):
                    if k != tor_psids:
                        for dim_ps in tor_psids:
                            if dim_ps in self.tt_ids_[k]:
                                elm_idx = self.tt_ids_[k].index(dim_ps)
                                ps_tups.append((k, dim_ps, lin_id + elm_idx))
                    elif k in tor_psids:
                        for eidx, val in enumerate(self.tt_ids_[k]):
                            ps_tups.append((k, val, lin_id + eidx))
                    lin_id += len(self.tt_ids_[k])
                return ps_tups
        else:
            pass

    def union(self, other):
        if isinstance(other, NeighborDOFs) and self.subset == other.subset:
            resultant = NeighborDOFs()
            resultant.set_subset(self.subset)
            resultant.n_bb_pairs_ = 0
            resultant.n_aa_pairs_ = 0
            resultant.n_tt_pairs_ = 0
            resultant.n_ba_pairs_ = 0
            resultant.n_bt_pairs_ = 0
            resultant.n_at_pairs_ = 0
            for k in sorted(self.bb_ids_.keys()):
                v = self.bb_ids_[k]
                if k not in other.bb_ids_:
                    resultant.bb_ids_[k] = np.array(sorted(v))
                    resultant.n_bb_pairs_ += len(v)
                else:
                    lst = list(v)
                    tmpl = sorted(
                        list(set(lst.extend(other.bb_ids_[k]))))
                    resultant.bb_ids_[k] = np.array(tmpl)
                    resultant.n_bb_pairs_ += len(tmpl)

            for k in sorted(self.aa_ids_.keys()):
                v = self.aa_ids_[k]
                if k not in other.aa_ids_:
                    resultant.aa_ids_[k] = np.array(sorted(v))
                    resultant.n_aa_pairs_ += len(v)
                else:
                    lst = list(v)
                    tmpl = sorted(
                        list(set(lst.extend(other.aa_ids_[k]))))
                    resultant.aa_ids_[k] = np.array(tmpl)
                    resultant.n_aa_pairs_ += len(tmpl)

            for k in sorted(self.tt_ids_.keys()):
                v = self.tt_ids_[k]
                if k not in other.tt_ids_:
                    resultant.tt_ids_[k] = np.array(sorted(v))
                    resultant.n_tt_pairs_ += len(v)
                else:
                    lst = list(v)
                    tmpl = sorted(
                        list(set(lst.extend(other.tt_ids_[k]))))
                    resultant.tt_ids_[k] = np.array(tmpl)
                    resultant.n_tt_pairs_ += len(tmpl)

            for k in sorted(self.ba_ids_.keys()):
                v = self.ba_ids_[k]
                if k not in other.ba_ids_:
                    resultant.ba_ids_[k] = np.array(sorted(v))
                    resultant.n_ba_pairs_ += len(v)
                else:
                    lst = list(v)
                    tmpl = sorted(
                        list(set(lst.extend(other.ba_ids_[k]))))
                    resultant.ba_ids_[k] = np.array(tmpl)
                    resultant.n_ba_pairs_ += len(tmpl)

            for k in sorted(self.bt_ids_.keys()):
                v = self.bt_ids_[k]
                if k not in other.bt_ids_:
                    resultant.bt_ids_[k] = np.array(sorted(v))
                    resultant.n_bt_pairs_ += len(v)
                else:
                    lst = list(v)
                    tmpl = sorted(
                        list(set(lst.extend(other.bt_ids_[k]))))
                    resultant.bt_ids_[k] = np.array(tmpl)
                    resultant.n_bt_pairs_ += len(tmpl)

            for k in sorted(self.at_ids_.keys()):
                v = self.at_ids_[k]
                if k not in other.at_ids_:
                    resultant.at_ids_[k] = np.array(sorted(v))
                    resultant.n_at_pairs_ += len(v)
                else:
                    lst = list(v)
                    tmpl = sorted(
                        list(set(lst.extend(other.at_ids_[k]))))
                    resultant.at_ids_[k] = np.array(tmpl)
                    resultant.n_at_pairs_ += len(tmpl)

            return resultant

        else:
            print(
                "WARNING:: union is defined for NeighborDOFs objects over same SubsetDOFs")
            return self.clone()

    def intersection(self, other):
        if isinstance(other, NeighborDOFs) and self.subset.equals(other.subset):
            resultant = NeighborDOFs()
            resultant.set_subset(self.subset)

            resultant.n_bb_pairs_ = 0
            resultant.n_aa_pairs_ = 0
            resultant.n_tt_pairs_ = 0
            resultant.n_ba_pairs_ = 0
            resultant.n_bt_pairs_ = 0
            resultant.n_at_pairs_ = 0
            for k, v in self.bb_ids_.items():
                if k in other.bb_ids_:
                    v1 = set(v)
                    v2 = set(other.bb_ids_[k])
                    v3 = np.array(sorted(list(v1.intersection(v2))))
                    resultant.bb_ids_[k] = v3
                    resultant.n_bb_pairs_ += len(v3)

            for k, v in self.aa_ids_.items():
                if k in other.aa_ids_:
                    v1 = set(v)
                    v2 = set(other.aa_ids_[k])
                    v3 = np.array(sorted(list(v1.intersection(v2))))
                    resultant.aa_ids_[k] = v3
                    resultant.n_aa_pairs_ += len(v3)

            for k, v in self.tt_ids_.items():
                if k in other.tt_ids_:
                    v1 = set(v)
                    v2 = set(other.tt_ids_[k])
                    v3 = np.array(sorted(list(v1.intersection(v2))))
                    resultant.tt_ids_[k] = v3
                    resultant.n_tt_pairs_ += len(v3)

            for k, v in self.ba_ids_.items():
                if k in other.ba_ids_:
                    v1 = set(v)
                    v2 = set(other.ba_ids_[k])
                    v3 = np.array(sorted(list(v1.intersection(v2))))
                    resultant.ba_ids_[k] = v3
                    resultant.n_ba_pairs_ += len(v3)

            for k, v in self.bt_ids_.items():
                if k in other.bt_ids_:
                    v1 = set(v)
                    v2 = set(other.bt_ids_[k])
                    v3 = np.array(sorted(list(v1.intersection(v2))))
                    resultant.bt_ids_[k] = v3
                    resultant.n_bt_pairs_ += len(v3)

            for k, v in self.at_ids_.items():
                if k in other.at_ids_:
                    v1 = set(v)
                    v2 = set(other.at_ids_[k])
                    v3 = np.array(sorted(list(v1.intersection(v2))))
                    resultant.at_ids_[k] = v3
                    resultant.n_at_pairs_ += len(v3)

            return resultant

        else:
            raise Exception(
                "ERROR:: intersection is defined for NeighborDOFs objects over same SubsetDOFs")
            return self.clone()

    def difference(self, other):
        if isinstance(other, NeighborDOFs) and self.subset == other.subset:
            res = NeighborDOFs()
            res.subset(self.subset)
            for k, v in self.bb_ids_.items():
                if k not in other.bb_ids_:
                    res.bb_ids_[k].update(v)

                else:
                    s1 = set(v.keys())
                    tmpl = sorted(
                        list(s1.difference(set(other.bb_ids_[k].keys()))))
                    res.bb_ids_[k] = np.array(tmpl)
            for k, v in self.aa_ids_.items():
                if k not in other.aa_ids_:
                    res.aa_ids_[k].update(v)

                else:
                    s1 = set(v.keys())
                    tmpl = sorted(
                        list(s1.difference(set(other.aa_ids_[k].keys()))))
                    res.aa_ids_[k] = np.array(tmpl)
            for k, v in self.tt_ids_.items():
                if k not in other.tt_ids_:
                    res.tt_ids_[k].update(v)

                else:
                    s1 = set(v.keys())
                    tmpl = sorted(
                        list(s1.difference(set(other.tt_ids_[k].keys()))))
                    res.tt_ids_[k] = np.array(tmpl)
            for k, v in self.ba_ids_.items():
                if k not in other.ba_ids_:
                    res.ba_ids_[k].update(v)

                else:
                    s1 = set(v.keys())
                    tmpl = sorted(
                        list(s1.difference(set(other.ba_ids_[k].keys()))))
                    res.ba_ids_[k] = np.array(tmpl)
            for k, v in self.bt_ids_.items():
                if k not in other.bt_ids_:
                    res.bt_ids_[k].update(v)

                else:
                    s1 = set(v.keys())
                    tmpl = sorted(
                        list(s1.difference(set(other.bt_ids_[k].keys()))))
                    res.bt_ids_[k] = np.array(tmpl)
            for k, v in self.at_ids_.items():
                if k not in other.at_ids_:
                    res.at_ids_[k].update(v)

                else:
                    s1 = set(v.keys())
                    tmpl = sorted(
                        list(s1.difference(set(other.at_ids_[k].keys()))))
                    res.at_ids_[k] = np.array(tmpl)
            return res

        else:
            print(
                "WARNING:: difference is defined for NeighborDOFs objects over same SubsetDOFs")
            return self.clone()

    def write(self, filename="neigh.txt"):
        msgs = []
        if self.subset is None:
            print("Set associated SubsetDOFs first then try to write again")
        elif isinstance(self.subset, subsetdofs.SubsetDOFs):
            nbe = self.subset.n_bnd_eff
            nae = self.subset.n_ang_eff
            nt = self.subset.n_tor_eff
            with open(filename, 'w') as fout:
                if self.bb_ids_:
                    nbb = nbe * (nbe - 1) / 2
                    nbb_s = 0
                    k_prev = -1
                    for k in sorted(self.bb_ids_.keys()):
                        if k - k_prev > 1:
                            for tmp in range(k_prev + 1, k):
                                fout.write("bond %d %d;\n" % (tmp, 0))
                        v = self.bb_ids_[k]
                        fout.write("bond %d %d " % (k, len(v)))
                        for vi in sorted(v):
                            if(vi > k):
                                fout.write(" " + str(vi))
                        fout.write(";\n")
                        nbb_s += len(v)
                        k_prev = k
                    fout.write("\n\n")
                    msg = "Cutoff bond-2d is {} of {} = {}% of full".format(nbb_s,
                                                                            nbb, round(nbb_s * 100.0 / nbb))
                    print(msg)
                    msgs.append(msg)

                if self.aa_ids_:
                    naa = nae * (nae - 1) / 2
                    naa_s = 0
                    k_prev = -1
                    for k in sorted(self.aa_ids_.keys()):
                        if k - k_prev > 1:
                            for tmp in range(k_prev + 1, k):
                                fout.write("angle %d %d;\n" % (tmp, 0))
                        v = self.aa_ids_[k]
                        fout.write("angle %d %d " % (k, len(v)))
                        for vi in sorted(v):
                            if(vi > k):
                                fout.write(" " + str(vi))
                        fout.write(";\n")
                        naa_s += len(v)
                        k_prev = k
                    fout.write("\n\n")
                    msg = "Cutoff angle-2d is {} of {} = {}% of full".format(naa_s,
                                                                             naa, round(naa_s * 100.0 / naa))
                    print(msg)
                    msgs.append(msg)

                if self.tt_ids_:
                    ntt = nt * (nt - 1) / 2
                    ntt_s = 0
                    k_prev = -1
                    for k in sorted(self.tt_ids_.keys()):
                        if k - k_prev > 1:
                            for tmp in range(k_prev + 1, k):
                                fout.write("torsion %d %d;\n" % (tmp, 0))
                        v = self.tt_ids_[k]
                        fout.write("torsion %d %d " % (k, len(v)))
                        for vi in sorted(v):
                            if(vi > k):
                                fout.write(" " + str(vi))
                        fout.write(";\n")
                        ntt_s += len(v)
                        k_prev = k
                    fout.write("\n\n")
                    msg = "Cutoff torsion-2d is {} of {} = {}% of full".format(ntt_s,
                                                                               ntt, round(ntt_s * 100.0 / ntt))
                    print(msg)
                    msgs.append(msg)

                if self.ba_ids_:
                    nba = nbe * nbe
                    nba_s = 0
                    k_prev = -1
                    for k in sorted(self.ba_ids_.keys()):
                        if k - k_prev > 1:
                            for tmp in range(k_prev + 1, k):
                                fout.write("bacross %d %d;\n" % (tmp, 0))
                        v = self.ba_ids_[k]
                        fout.write("bacross %d %d " % (k, len(v)))
                        for vi in sorted(v):
                            fout.write(" " + str(vi))
                        fout.write(";\n")
                        nba_s += len(v)
                        k_prev = k
                    fout.write("\n\n")
                    msg = "Cutoff bond-angle-cross is {} of {} = {}% of full".format(nba_s,
                                                                                     nba, round(nba_s * 100.0 / nba))
                    print(msg)
                    msgs.append(msg)

                if self.bt_ids_:
                    nbt = nbe * nt
                    nbt_s = 0
                    k_prev = -1
                    for k in sorted(self.bt_ids_.keys()):
                        if k - k_prev > 1:
                            for tmp in range(k_prev + 1, k):
                                fout.write("bdcross %d %d;\n" % (tmp, 0))
                        v = self.bt_ids_[k]
                        fout.write("bdcross %d %d " % (k, len(v)))
                        for vi in sorted(v):
                            # if(vi > k):
                            fout.write(" " + str(vi))
                        fout.write(";\n")
                        nbt_s += len(v)
                        k_prev = k
                    fout.write("\n\n")
                    msg = "Cutoff bond-torsion-cross is {} of {} = {}% of full".format(nbt_s,
                                                                                       nbt, round(nbt_s * 100.0 / nbt))
                    print(msg)
                    msgs.append(msg)

                if self.at_ids_:
                    nat = nae * nt
                    nat_s = 0
                    k_prev = -1
                    for k in sorted(self.at_ids_.keys()):
                        if k - k_prev > 1:
                            for tmp in range(k_prev + 1, k):
                                fout.write("adcross %d %d;\n" % (tmp, 0))
                        v = self.at_ids_[k]
                        fout.write("adcross %d %d " % (k, len(v)))
                        for vi in sorted(v):
                            # if(vi > k):
                            fout.write(" " + str(vi))
                        fout.write(";\n")
                        nat_s += len(v)
                        k_prev = k
                    fout.write("\n\n")
                    msg = "Cutoff angle-torsion-cross is {} of {} = {}% of full".format(nat_s,
                                                                                        nat, round(nat_s * 100.0 / nat))
                    print(msg)
                    msgs.append(msg)
        return "\n".join(msgs)


    def load(self, fname, entropy_workset=BATSet.XY2D):
        if self.subset is not None and isinstance(entropy_workset, BATSet):
            with open(fname, 'r') as fin:
                for ln in utls.readlines_delim(fin, ";"):
                    words = ln.split()
                    if len(words) > 3:
                        dtype = words[0]
                        dim_id = int(words[1])
                        n_neighs = int(words[2])
                        if n_neighs == 0:
                            continue
                        if dtype == 'bond' and not entropy_workset.has(BATSet.BB2D):
                            continue
                        if dtype == 'angle' and not entropy_workset.has(BATSet.AA2D):
                            continue
                        if dtype == 'torsion' and not entropy_workset.has(BATSet.DD2D):
                            continue
                        if dtype == 'bacross' and not entropy_workset.has(BATSet.BA2D):
                            continue
                        if dtype == 'bdcross' and not entropy_workset.has(BATSet.BD2D):
                            continue
                        if dtype == 'adcross' and not entropy_workset.has(BATSet.AD2D):
                            continue
                        dims_neighs = list(map(int, words[3:]))
                        n_reads = len(dims_neighs)
                        if n_neighs != n_reads:
                            raise Exception("Error in reading (%s) neighbors of %s(%s), found only %s" % (
                                str(n_neighs), dtype, str(dim_id), str(n_reads)))
                        if dtype == 'bond':
                            self.bb_ids_[dim_id] = np.array(sorted(dims_neighs))
                            self.n_bb_pairs_ += len(dims_neighs)
                        elif dtype == 'angle':
                            self.aa_ids_[dim_id] = np.array(sorted(dims_neighs))
                            self.n_aa_pairs_ += len(dims_neighs)
                        elif dtype == 'torsion':
                            self.tt_ids_[dim_id] = np.array(sorted(dims_neighs))
                            self.n_tt_pairs_ += len(dims_neighs)
                        elif dtype == 'bacross':
                            self.ba_ids_[dim_id] = np.array(sorted(dims_neighs))
                            self.n_ba_pairs_ += len(dims_neighs)
                        elif dtype == 'bdcross':
                            self.bt_ids_[dim_id] = np.array(sorted(dims_neighs))
                            self.n_bt_pairs_ += len(dims_neighs)
                        elif dtype == 'adcross':
                            self.at_ids_[dim_id] = np.array(sorted(dims_neighs))
                            self.n_at_pairs_ += len(dims_neighs)
        else:
            print("Please set corresponding SubsetDOFs first and try again..")
