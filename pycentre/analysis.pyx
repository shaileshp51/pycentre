from __future__ import print_function
from copy import deepcopy

import os
import sys
import csv
import math
import pickle
import textwrap
import collections

import networkx as nx
import pytraj as pt
import numpy as np
import netCDF4 as nc

cimport numpy as np

try:
    from pycentre import utils_ as utls
    from pycentre import centretypes
    from pycentre import subsetdofs
    from pycentre import neighbordofs
    from pycentre.datatypes import *

except ImportError:
    try:
        from . import utils_ as utls
        from . import centretypes
        from . import subsetdofs
        from . import neighbordofs
        from .datatypes import *
    except ImportError as err:
        raise ("Error import: " + str(err))


INFINITE_NEIGHBOR_CUTOFF = 999999

class ConverganceReport:
    """
    A class to represent a ConverganceReport of entropy estimation results.

    This class represents a convergence report for entropy estimation,
        calculated from program ``CENTRE``

    Parameters
    ----------
    filename : str
        It can be ``None`` or a ``str`` representing a valid absolute filename

    Attributes
    ----------
    filename : ``str`` default=``None``
        It holds name of the associated file with the report, it is optional
    columns : list[str]
        A list of column names of the report. It is immutable.
    report : ``OrderedDict``
        An ordered dict to hold report rows data, columns are used as keys
        and an associted list[float] as value for every column.

    """

    def __init__(self, filename=None):
        """
        Construct a ``ConverganceReport`` object.

        """
        self.filename = filename
        self.columns = [
            'TIME',    'BOND-S1',  'ANGLE-S1', 'DIHED-S1', 'B/B-S2',
            'B/B-I2',  'A/A-S2',   'A/A-I2',   'D/D-S2',   'D/D-I2',
            'B/A-S2',  'B/A-I2',   'B/D-S2',   'B/D-I2',   'A/D-S2',
            'A/D-I2',  'S1',       'S2'
        ]
        self.report = collections.OrderedDict()
        for hd in self.columns:
            self.report[hd] = []

    def get_row_odict(self):
        """
        Create and return an OrderedDict object as row-placeholder of report

        Returns
        -------
        An OrderedDict object with report columns as key and value 0.0

        """
        obj_row = collections.OrderedDict()
        for hd in self.columns:
            obj_row[hd] = 0.0
        return obj_row

    def add_row(self, obj_row):
        """
        Add supplied row object to the report

        It receives an OrderedDict object representing a row of the report,
        and adds this row to the report.

        Parameters
        ----------
        obj_row : OrderedDict
            object with columns of the report as keys, and corresponding
            values for the columns.

        Notes
        -----
            It mutates state of the ``object``.

        """
        for hd in obj_row.items():
            self.report[hd[0]].append(hd[1])

    def get_cell(self, int row_index, col_name):
        """
        Get value of specified column of report for row with given row_index

        Parameters
        ----------
        row_index : int
            Index of row in report whose column value is desired. It can be a
            ``+ve/-ve int`` in interval ``(-num_rows, num_rows)``, where
            ``num_row`` is the total number of rows in the ``ConverganceReport``
            object.
            None is returned if row_index does not exist and
        col_name : str
            A column existing in the report object

        Returns
        -------
        float or None
            A float value of the cell for valid input, otherwise None

        """
        nrows = len(self.report['TIME'])
        if abs(row_index) < nrows:
            if col_name in self.columns:
                return self.report[col_name][row_index]
            else:
                print(
                    "ERROR:: Request col_name({0}) does not exist.".format(col_name))
        else:
            print("ERROR:: Request row_index({0}) does not exist. Total rows is: {1}".format(
                row_index, nrows))
        return None

    def get_row_values(self, int row_index):
        """
        Get values of all the column of report for row with given row_index

        Parameters
        ----------
        row_index : int
            Index of row in report whose column value is desired. It can be a
            ``+ve/-ve int`` in interval ``(-num_rows, num_rows)``, where
            ``num_row`` is the total number of rows in the ``ConverganceReport``
            object.
            None is returned if row_index does not exist and

        Returns
        -------
        OrderedDict or None
            A OrderedDict object with keys as columns of the report and values
            as associated column values for the given row_index.
            None is return for invalid row_index.

        """
        nrows = len(self.report['TIME'])
        if abs(row_index) < nrows:
            obj_row = collections.OrderedDict()
            for hd in self.columns:
                obj_row[hd] = self.report[hd][row_index]
            return obj_row
        else:
            print("ERROR:: Request row_index({0}) does not exist. Total rows is: {1}".format(
                row_index, nrows))
        return None

    def get_column(self, col_name):
        """
        Get values in all rows of the request column of report.

        Parameters
        ----------
        col_name : str
            A column existing in the report object

        Returns
        -------
        list of float
            A list of float values for the request column of the report.
            list values arranged corresponding to row_indices' increasing-sort order.

            None if column does not exist.

        """
        if col_name in self.columns:
            return self.report[col_name]
        else:
            print(
                "ERROR:: Request col_name({0}) does not exist.".format(col_name))
        return None

    def write(self, filename=None, round_digits=4, col_width=13):
        """
        Save/Write the report in ascii-text file.

        It writes out the report in an ascii-text file with provided filename
        or filename ``field`` of the object if None provided. If both are None
        print ``warning`` and do nothing.

        Parameters
        ----------
        filename : ``str`` default=``None``
            An filename for writing the ``ConverganceReport``. If filename is
            not provided but ``ConverganceReport`` object has filename field
            set it will be used for writing object, otherwise just a warning
            will be issued without writing data.

        round_digits : ``int`` default=4
            Number of digits to print in report after decimal point
        col_width : ``int`` default=13
            Field-width for column printing, each column will be left-aligned
            in print and fields will be comma-seperated.

        """
        fname = filename if filename is not None else self.filename
        cdef int i
        if fname is None:
            print("WARNING:: a filename is expected to write. try with a valid filename")
            return
        else:
            fmt = "{0:<" + str(col_width) + "}"
            with open(fname, 'w') as fout:
                fout.write(", ".join([fmt.format(cl)
                                      for cl in self.columns]) + '\n')
                nrows = len(self.report['TIME'])
                for i in range(nrows):
                    tmp = []
                    for k, v in self.report.items():
                        val = round(v[i], 4) if k != 'TIME' else int(v[i])
                        tmp.append(fmt.format(val))
                    fout.write(", ".join(tmp) + '\n')

    def read(self, filename=None):
        """
        Read and add rows to the report from ascii-text file.

        It reads the report from an ascii-text file with provided filename
        or filename ``field`` of the object if None provided. If both are None
        print ``warning`` and do nothing.

        Parameters
        ----------
        filename : ``str`` default=``None``
            An filename to read from ``ConverganceReport``. If filename is
            not provided but ``ConverganceReport`` object has filename field
            set it will be used for reading object, otherwise just a warning
            will be issued without reading data.

        """
        filename_ = self.filename if filename is None else filename
        if filename_ is None or not os.path.exists(filename_):
            print("Please provide a valid filename to read from, provided: {0}".format(
                filename_))
        else:
            with open(filename_, 'r') as fin:
                fin.readline()
                # Erase all existing values and read from file
                for cl in self.columns:
                    self.report[cl] = []
                for ln in fin.readlines():
                    words = ln.strip().split(',')
                    # print(words)
                    for wi, wr in enumerate(words):
                        self.report[self.columns[wi]].append(float(wr.strip()))


def write_MIST(filename, edge_set):
    """
    Write supplied MIST edges to given filename.

    It writes MIST in numeric-descending order of values of ``mi`` key of edge
    property ``dict`` 3rd entry of edge (3-tuple). It has comma seperated
    5-columns. each edge in one-row.

    Parameters
    ----------
    filename : ``str``
        A filename to write into.
    edge_set : edge_set[3-tuple(from-node, to-node, prorerty-dict)]

    """
    cols = ['TD1', 'TD2', 'IndexDim1', 'IndexDim2', 'MI']
    with open(filename, 'w') as fout:
        fout.write(', '.join(cols) + '\n')
        formatstr = "{0:<3}, {1:<3}, {2:<9}, {3:<9}, {4}\n"
        for e in sorted(edge_set.edges(data=True), key=lambda x: x[2]['mi'], reverse=True):
            fout.write(formatstr.format(e[0][0], e[1][0], e[0][1:], e[1][1:],
                                        round(e[2]['mi'], 6)
                                        )
                )


def get_mi_vector(np.ndarray data1d1, np.ndarray data1d2, np.ndarray data2d,
                  isubset, neigh_current, char * which2d, int gengraph, double threshold):
    """
    Calculate values of mutual-information terms.

    Parameters
    ----------
    data1d1 : ``numpy.ndarray[dtype=float, dims=1]``
        An array of 1st order entropy values for the first dimension-set of
        dimension/dimension pair identified by ``param which2d``.
        It length must be equlal to the number of elements of type
        1st-dimension-set of pair ``param which2d`` in the given subset
        object supplied with ``param isubset``.
    data1d2 : ``numpy.ndarray[dtype=float, dims=1]``
        An array of 1st order entropy values for the second dimenstion-set of
        dimension/dimension pair identified by ``param which2d``.
        It length must be equlal to the number of elements of type
        2nd-dimension-set of pair ``param which2d`` in the given subset
        object supplied with ``param isubset``.
    data2d : ``numpy.ndarray[dtype=float, dims=1]``
        An array of 2nd order entropy values for the dimension/dimension pair
        identified by ``param which2d``.
        It should satisfy following conditions.
        1. Its length should be greater-or-equal to the length of list of type
         ``param which2d`` of supplied neighbor object.
            a) Equality holds if scoring is being done for the same neighbor-set
               for with entropy calculation was carried through CENTRE.
            b) Inequality holds if scoring is being done for a neighborhood with
               smaller radius than one with which entropy calculation was performed.
    isubset : SubsetDOFs
        A SubsetDOFs which holds information of DOFs subset considered for
        entropy estimation
    ineigh : NeighborDOFs
        A NeighborDOFs object which holds information of DOFs falling in
        neighborhood of given radius.
        This neighborhood supplied can be one with which entropy calculation
        was performed or a neighborhood with smaller radius, because smaller
        neighborhood is strictly a subset of neighborhood of larger radius.
    which2d : char* or byte-array
        It represents DOF-type pair b/a/t standing for bond/angle/torsion
        respectively. It has two chars one for first DOF-type and other for
        second DOF-type. It can be one of {'bb', 'ba', 'bt', 'aa', 'at', 'tt'}
    gengraph : int
        0=False nonzero True
        An interger used as boolean to flag whether geneterate a ``networkx.Graph``
        object from Mutual-Information (MI) data. Here DOFs involved will be
        used as nodes and MI value for DOF-pair or node-pair as edge-weight.
        Edge-property-dict has two keys 'mi' & '-mi' with MI & -MI as weights,
        here '-mi' is used for calculation of MIST by calculating minimum-
        spanning-tree of graph over negative weights of edges it becomes.
        Maximum-Infomration-Spanning-Tree.
    threshold : double
        A real thresold value, pairs with MI  <= thresold will be considered
        zero and will not be included in graph-edge creation and MI scoring.

    """
    mi_data = np.zeros(data2d.shape[0])
    cdef int i
    cdef int idn1, idn2
    cdef int ids1, ids2

    edges = []
    if which2d == b'bb':
        if len(data2d) >= neigh_current.n_bb_pairs_:
            for idn1 in sorted(neigh_current.bb_ids_.keys()):
                for idn2 in neigh_current.bb_ids_[idn1]:
                    if idn1 in isubset.bnd_ids_ and idn2 in isubset.bnd_ids_:
                        ids1, ids2 = (isubset.bnd_ids_[
                                      idn1], isubset.bnd_ids_[idn2])
                        i = neigh_current.idpair2indexmap.idpair2index(which2d, idn1, idn2)
                        mi_data[i] = data1d1[ids1] + data1d2[ids2] - data2d[i]
                        if mi_data[i] <= threshold:
                            mi_data[i] = 0.0
                        if gengraph and (mi_data[i] > threshold):
                            edges.append(
                                ('B' + str(ids1), 'B' + str(ids2), mi_data[i]))
                    else:
                        print('b', end='', flush=True)

            if not gengraph:
                sum_mis = (np.sum(data2d), np.sum(mi_data))
        else:
            msg = "ERROR:: Input[{0}] neighbors-count({1}) and present-in-file({2}) are unequal"
            print(msg.format(which2d, len(neigh_current.n_bb_pairs_),  len(data2d)))
    if which2d == b'aa':
        if len(data2d) >= neigh_current.n_aa_pairs_:
            for idn1 in sorted(neigh_current.aa_ids_.keys()):
                for idn2 in neigh_current.aa_ids_[idn1]:
                    if idn1 in isubset.ang_ids_ and idn2 in isubset.ang_ids_:
                        ids1, ids2 = (isubset.ang_ids_[
                                      idn1], isubset.ang_ids_[idn2])
                        i = neigh_current.idpair2indexmap.idpair2index(which2d, idn1, idn2)
                        mi_data[i] = data1d1[ids1] + data1d2[ids2] - data2d[i]
                        if mi_data[i] <= threshold:
                            mi_data[i] = 0.0
                        if gengraph and (mi_data[i] > threshold):
                            edges.append(
                                ('A' + str(ids1), 'A' + str(ids2), mi_data[i]))
                    else:
                        print('a', end='', flush=True)
            if not gengraph:
                sum_mis = (np.sum(data2d), np.sum(mi_data))
        else:
            msg = "ERROR:: Input[{0}] neighbors-count({1}) and present-in-file({2}) are unequal"
            print(msg.format(which2d, len(neigh_current.n_aa_pairs_),  len(data2d)))
    if which2d == b'tt':
        if len(data2d) >= neigh_current.n_tt_pairs_:
            for idn1 in sorted(neigh_current.tt_ids_.keys()):
                for idn2 in neigh_current.tt_ids_[idn1]:
                    if idn1 in isubset.tor_ids_ and idn2 in isubset.tor_ids_:
                        ids1, ids2 = (isubset.tor_ids_[
                                      idn1], isubset.tor_ids_[idn2])
                        i = neigh_current.idpair2indexmap.idpair2index(which2d, idn1, idn2)
                        mi_data[i] = data1d1[ids1] + data1d2[ids2] - data2d[i]
                        if mi_data[i] <= threshold:
                            mi_data[i] = 0.0
                        if gengraph and (mi_data[i] > threshold):
                            edges.append(
                                ('T' + str(ids1), 'T' + str(ids2), mi_data[i]))
                    else:
                        print('a', end='', flush=True)
            if not gengraph:
                sum_mis = (np.sum(data2d), np.sum(mi_data))
        else:
            msg = "ERROR:: Input[{0}] neighbors-count({1}) and present-in-file({2}) are unequal"
            print(msg.format(which2d, len(neigh_current.n_tt_pairs_),  len(data2d)))
    if which2d == b'ba':
        if len(data2d) >= neigh_current.n_ba_pairs_:
            for idn1 in sorted(neigh_current.ba_ids_.keys()):
                for idn2 in neigh_current.ba_ids_[idn1]:
                    if idn1 in isubset.bnd_ids_ and idn2 in isubset.ang_ids_:
                        ids1, ids2 = (isubset.bnd_ids_[
                                      idn1], isubset.ang_ids_[idn2])
                        i = neigh_current.idpair2indexmap.idpair2index(which2d, idn1, idn2)
                        mi_data[i] = data1d1[ids1] + data1d2[ids2] - data2d[i]
                        if mi_data[i] <= threshold:
                            mi_data[i] = 0.0
                        if gengraph and (mi_data[i] > threshold):
                            edges.append(
                                ('B' + str(ids1), 'A' + str(ids2), mi_data[i]))
                    else:
                        print('x', end='', flush=True)
            if not gengraph:
                sum_mis = (np.sum(data2d), np.sum(mi_data))
        else:
            msg = "ERROR:: Input[{0}] neighbors-count({1}) and present-in-file({2}) are unequal"
            print(msg.format(which2d, len(neigh_current.n_ba_pairs_),  len(data2d)))
    if which2d == b'bt':
        if len(data2d) >= neigh_current.n_bt_pairs_:
            for idn1 in sorted(neigh_current.bt_ids_.keys()):
                for idn2 in neigh_current.bt_ids_[idn1]:
                    if idn1 in isubset.bnd_ids_ and idn2 in isubset.tor_ids_:
                        ids1, ids2 = (isubset.bnd_ids_[
                                      idn1], isubset.tor_ids_[idn2])
                        i = neigh_current.idpair2indexmap.idpair2index(which2d, idn1, idn2)
                        mi_data[i] = data1d1[ids1] + data1d2[ids2] - data2d[i]
                        if mi_data[i] <= threshold:
                            mi_data[i] = 0.0
                        if gengraph and mi_data[i] > threshold:
                            edges.append(
                                ('B' + str(ids1), 'T' + str(ids2), mi_data[i]))
                    else:
                        print('y', end='', flush=True)
            if not gengraph:
                sum_mis = (np.sum(data2d), np.sum(mi_data))
        else:
            msg = "ERROR:: Input[{0}] neighbors-count({1}) and present-in-file({2}) are unequal"
            print(msg.format(which2d, len(neigh_current.n_bt_pairs_),  len(data2d)))
    if which2d == b'at':
        if len(data2d) >= neigh_current.n_at_pairs_:
            for idn1 in sorted(neigh_current.at_ids_.keys()):
                for idn2 in neigh_current.at_ids_[idn1]:
                    if idn1 in isubset.ang_ids_ and idn2 in isubset.tor_ids_:
                        ids1, ids2 = (isubset.ang_ids_[
                                      idn1], isubset.tor_ids_[idn2])
                        i = neigh_current.idpair2indexmap.idpair2index(which2d, idn1, idn2)
                        mi_data[i] = data1d1[ids1] + data1d2[ids2] - data2d[i]
                        if mi_data[i] <= threshold:
                            mi_data[i] = 0.0
                        if gengraph and mi_data[i] > threshold:
                            edges.append(
                                ('A' + str(ids1), 'T' + str(ids2), mi_data[i]))
                    else:
                        print('y', end='', flush=True)
            if not gengraph:
                sum_mis = (np.sum(data2d), np.sum(mi_data))
        else:
            msg = "ERROR:: Input[{0}] neighbors-count({1}) and present-in-file({2}) are unequal"
            print(msg.format(which2d, len(neigh_current.n_bt_pairs_),  len(data2d)))
    if gengraph:
        return mi_data, edges

    return mi_data, sum_mis


def cal_conv_report_row(inputs, ent_contri_files, estimator_id, binscheme_id,
                        step_id, neigh_current, out_odict, threshold, gengraph=False,
                        writemist=False, mistfname='', debug_state=False):
    """
    Calculate a row of ConverganceReport for parameters set from output files
    of CENTRE entropy calculation.

    Parameters
    ----------
    inputs : CentreInputs
        An object holding input parametres used for entropy calculation using
        CENTRE
    ent_contri_files : ``dict {str: netcdf-file-handle-opend-for-reading}``
        A dictionary object holding entropy-calculation performed for DOF sets
        as key and file handles to associated entropy-contribution as values.
    estimator_id : ``int``
        Index(0-based) of estimator in CentreInputs.estimators for which report
        is desired.
    binscheme_id : ``int``
        Index(0-based) of binscheme in CentreInputs.bats.binschemes for which
        report is desired.
    step_id : ``int``
        Index(0-based) of step-index in [0..CentreInputs.control.nstep] for which
        report is desired.
    ineigh : ``NeighborDOFs``
        A NeighborDOFs object which holds information of DOFs falling in
        neighborhood of given radius.
        This neighborhood supplied can be one with which entropy calculation
        was performed or a neighborhood with smaller radius, because smaller
        neighborhood is strictly a subset of neighborhood of larger radius.
    out_odict : ``ConverganceReport`` row-dict
        It is a out parameter and calculated values will be updated to corresponding
        columns of the row as result of execution of function.
    threshold : ``double``
        A real thresold value, pairs with MI  <= thresold will be considered
        zero and will not be included in graph-edge creation and MI scoring.
    gengraph : ``boolean`` default=``False``
        A boolean to flag wheter geneterate a ``networkx.Graph``
        object from Mutual-Information (MI) data. Here DOFs involved will be
        used as nodes and MI value for DOF-pair or node-pair as edge-weight.
        Edge-property-dict has two keys 'mi' & '-mi' with MI & -MI as weights,
        here '-mi' is used for calculation of MIST by calculating minimum-
        spanning-tree of graph over negative weights of edges it becomes.
        Maximum-Infomration-Spanning-Tree.
    writemist : ``boolean`` default=``False``
        whether to write MIST to a file with name provided by mistfname
    mistfname : ``str`` default=''
        filename for writing MIST tree.
    debug_state: ``boolean`` default=``False``

    Returns
    -------
    s1_dict : a dict of 1D entropy contributions
    graph   : a networkx graph object with interacting DFs as nodes and mi
        as the edge weight

    """
    edges_list = []
    s1_dict = {}
    workset = inputs.entropy.workset
    if workset.has(centretypes.BATSet.B1D) or workset.has(centretypes.BATSet.BB2D):
        sb_1d = ent_contri_files['b'].variables['contri_S'][:,
                                                            estimator_id, binscheme_id, step_id]
        out_odict['BOND-S1'] = np.sum(sb_1d)
        for _id, _idx in inputs.subset.bnd_ids_.items():
            s1_dict['B' + str(_id)] = sb_1d[_idx]
    if workset.has(centretypes.BATSet.A1D) or workset.has(centretypes.BATSet.AA2D):
        sa_1d = ent_contri_files['a'].variables['contri_S'][:,
                                                            estimator_id, binscheme_id, step_id]
        out_odict['ANGLE-S1'] = np.sum(sa_1d)
        for _id, _idx in inputs.subset.ang_ids_.items():
            s1_dict['A' + str(_id)] = sa_1d[_idx]
    if workset.has(centretypes.BATSet.D1D) or workset.has(centretypes.BATSet.DD2D):
        sd_1d = ent_contri_files['t'].variables['contri_S'][:,
                                                            estimator_id, binscheme_id, step_id]
        out_odict['DIHED-S1'] = np.sum(sd_1d)
        for _id, _idx in inputs.subset.tor_ids_.items():
            s1_dict['T' + str(_id)] = sd_1d[_idx]
    if workset.has(centretypes.BATSet.BB2D):
        sbb_2d = ent_contri_files['bb'].variables['contri_S'][:,
                                                              estimator_id, binscheme_id, step_id]
        if gengraph:
            bb_mi2d, edges = get_mi_vector(
                sb_1d, sb_1d, sbb_2d, inputs.subset, neigh_current, b'bb', gengraph, threshold)
            edges_list.extend(edges)
            out_odict['B/B-S2'] = np.sum(sbb_2d)
            if debug_state:
                print("{} has {} pairs".format("B/B", len(edges)), flush=True)
        else:
            bb_mi2d, sum_mi = get_mi_vector(
                sb_1d, sb_1d, sbb_2d, inputs.subset, neigh_current, b'bb', gengraph, threshold)
            out_odict['B/B-S2'], out_odict['B/B-I2'] = sum_mi
    if workset.has(centretypes.BATSet.AA2D):
        saa_2d = ent_contri_files['aa'].variables['contri_S'][:,
                                                              estimator_id, binscheme_id, step_id]
        if gengraph:
            aa_mi2d, edges = get_mi_vector(
                sa_1d, sa_1d, saa_2d, inputs.subset, neigh_current, b'aa', gengraph, threshold)
            edges_list.extend(edges)
            out_odict['A/A-S2'] = np.sum(saa_2d)
            if debug_state:
                print("{} has {} pairs".format("A/A", len(edges)), flush=True)
        else:
            aa_mi2d, sum_mi = get_mi_vector(
                sa_1d, sa_1d, saa_2d, inputs.subset, neigh_current, b'aa', gengraph, threshold)
            out_odict['A/A-S2'], out_odict['A/A-I2'] = sum_mi
    if workset.has(centretypes.BATSet.DD2D):
        stt_2d = ent_contri_files['tt'].variables['contri_S'][:,
                                                              estimator_id, binscheme_id, step_id]
        if gengraph:
            tt_mi2d, edges = get_mi_vector(
                sd_1d, sd_1d, stt_2d, inputs.subset, neigh_current, b'tt', gengraph, threshold)
            edges_list.extend(edges)
            out_odict['D/D-S2'] = np.sum(stt_2d)
            if debug_state:
                print("{} has {} pairs".format("T/T", len(edges)), flush=True)
        else:
            tt_mi2d, sum_mi = get_mi_vector(
                sd_1d, sd_1d, stt_2d, inputs.subset, neigh_current, b'tt', gengraph, threshold)
            out_odict['D/D-S2'], out_odict['D/D-I2'] = sum_mi
    if workset.has(centretypes.BATSet.B1D) > 0 and workset.has(centretypes.BATSet.A1D) and workset.has(centretypes.BATSet.BA2D):
        sba_2d = ent_contri_files['ba'].variables['contri_S'][:,
                                                              estimator_id, binscheme_id, step_id]
        if gengraph:
            ba_mi2d, edges = get_mi_vector(
                sb_1d, sa_1d, sba_2d, inputs.subset, neigh_current, b'ba', gengraph, threshold)
            edges_list.extend(edges)
            out_odict['B/A-S2'] = np.sum(sba_2d)
            if debug_state:
                print("{} has {} pairs".format("B/A", len(edges)), flush=True)
        else:
            ba_mi2d, sum_mi = get_mi_vector(
                sb_1d, sa_1d, sba_2d, inputs.subset, neigh_current, b'ba', gengraph, threshold)
            out_odict['B/A-S2'], out_odict['B/A-I2'] = sum_mi
    if workset.has(centretypes.BATSet.B1D) and workset.has(centretypes.BATSet.D1D) and workset.has(centretypes.BATSet.BD2D):
        sbt_2d = ent_contri_files['bt'].variables['contri_S'][:,
                                                              estimator_id, binscheme_id, step_id]
        if gengraph:
            bt_mi2d, edges = get_mi_vector(
                sb_1d, sd_1d, sbt_2d, inputs.subset, neigh_current, b'bt', gengraph, threshold)
            edges_list.extend(edges)
            out_odict['B/D-S2'] = np.sum(sbt_2d)
            if debug_state:
                print("{} has {} pairs".format("B/T", len(edges)), flush=True)
        else:
            bt_mi2d, sum_mi = get_mi_vector(
                sb_1d, sd_1d, sbt_2d, inputs.subset, neigh_current, b'bt', gengraph, threshold)
            out_odict['B/D-S2'], out_odict['B/D-I2'] = sum_mi
    if workset.has(centretypes.BATSet.A1D) and workset.has(centretypes.BATSet.D1D) and workset.has(centretypes.BATSet.AD2D):
        sat_2d = ent_contri_files['at'].variables['contri_S'][:,
                                                              estimator_id, binscheme_id, step_id]
        if gengraph:
            at_mi2d, edges = get_mi_vector(
                sa_1d, sd_1d, sat_2d, inputs.subset, neigh_current, b'at', gengraph, threshold)
            edges_list.extend(edges)
            out_odict['A/D-S2'] = np.sum(sat_2d)
            if debug_state:
                print("{} has {} pairs".format("A/T", len(edges)), flush=True)
        else:
            at_mi2d, sum_mi = get_mi_vector(
                sa_1d, sd_1d, sat_2d, inputs.subset, neigh_current, b'at', gengraph, threshold)
            out_odict['A/D-S2'], out_odict['A/D-I2'] = sum_mi
    graph = None
    if gengraph:
        graph = nx.Graph()
        graph.add_weighted_edges_from(edges_list, weight='mi')
        mist = nx.maximum_spanning_tree(graph, weight='mi')
        mist_edges = sorted(mist.edges(data=True), key=lambda x: x[2]['mi'], reverse=True)
        if gengraph and writemist:
            write_MIST(mistfname, mist)
        out_odict['B/B-I2'] = np.sum([e[2]['mi']
                                      for e in mist_edges if e[0][0] == 'B' and e[1][0] == 'B'])
        out_odict['A/A-I2'] = np.sum([e[2]['mi']
                                      for e in mist_edges if e[0][0] == 'A' and e[1][0] == 'A'])
        out_odict['D/D-I2'] = np.sum([e[2]['mi']
                                      for e in mist_edges if e[0][0] == 'T' and e[1][0] == 'T'])
        out_odict['B/A-I2'] = np.sum([e[2]['mi']
                                      for e in mist_edges if ((e[0][0] == 'B' and e[1][0] == 'A') or (e[0][0] == 'A' and e[1][0] == 'B'))])
        out_odict['B/D-I2'] = np.sum([e[2]['mi']
                                      for e in mist_edges if ((e[0][0] == 'B' and e[1][0] == 'T') or (e[0][0] == 'T' and e[1][0] == 'B'))])
        out_odict['A/D-I2'] = np.sum([e[2]['mi']
                                      for e in mist_edges if ((e[0][0] == 'A' and e[1][0] == 'T') or (e[0][0] == 'T' and e[1][0] == 'A'))])
        #print(len(mist_edges), mist_edges)
    out_odict['S1'] = out_odict['BOND-S1'] + \
        out_odict['ANGLE-S1'] + out_odict['DIHED-S1']
    out_odict['S2'] = out_odict['S1'] - out_odict['B/B-I2'] - \
        out_odict['A/A-I2'] - out_odict['D/D-I2'] - out_odict['B/A-I2'] - \
        out_odict['B/D-I2'] - out_odict['A/D-I2']

    return s1_dict, graph


def to_int(word):
    """
    Convert input to integer equivallent if possible

    Parameters
    ---------
    word : ``str``
        a decimal-numeric string

    Returns
    -------
    ``int`` or ``None``
    None if word is not a valid decimal-numeric string

    """
    try:
        return int(word)
    except:
        return None


def str2range(rangestring='first:last:10', first=0, last=40):
    """
    Convert range_string to list

    Parameters
    ----------
    rangestring : ``str``
        A range string of format [start:stop):step, stop not included
        first-last keywords supported, default: first=0, default: last=40
    first : ``int`` default=0
        interger corresponding to keyword first i.e. start value of range
    last : ``int`` default=40
        integer corresponding to keyword last i.e. stop

    Returns
    -------
    ``list[int]``
        Return a list for the range string

    """
    writemiststeps_rng = []
    wordwmist = rangestring.split(':')
    if len(wordwmist) > 3:
        raise Exception('Invalid range-string')
    if len(wordwmist) == 3:
        wordwmist[2].strip()
        if to_int(wordwmist[2]) is not None:
            writemiststeps_rng.insert(0, to_int(wordwmist[2]))
        else:
            writemiststeps_rng.insert(0, last)
    if len(wordwmist) >= 2:
        wordwmist[1].strip()
        if wordwmist[1] in ['last']:
            writemiststeps_rng.insert(0, last)
        elif to_int(wordwmist[1]) is not None:
            writemiststeps_rng.insert(0, to_int(wordwmist[1]))
        else:
            writemiststeps_rng.insert(0, last)
    if len(wordwmist) >= 1:
        wordwmist[0].strip()
        if wordwmist[0] in ['first']:
            writemiststeps_rng.insert(0, first)
        elif wordwmist[0] in ['last']:
            writemiststeps_rng.insert(0, last)
        elif to_int(wordwmist[0]) is not None:
            writemiststeps_rng.insert(0, to_int(wordwmist[0]))
        else:
            writemiststeps_rng.insert(0, last)
    return [i for i in range(*writemiststeps_rng)]


def make_conv_report(inputs, moltree, cutoffs, neigh_fprefix,
                     scoring_method=centretypes.ScoringMethods.MIE,
                     foretimators='all', forbinschemes='all', forsteps='all',
                     writemiststeps='9:last:10', gengraph=False,
                     threshold=1.0e-8, outfileext='txt', debug_state=False):
    """
    Prepare ConverganceReport for parameters set from output files
    of CENTRE entropy calculation.

    Parameters
    ----------
    inputs : ``CentreInputs``
        An object holding input parametres used for entropy calculation using
        CENTRE
    moltree : ``MMTree``
        A MMTree object constructed from molecule structure
    cutoff : ``list`` [int]
        A list of integer of cutoff values
    neigh_fprefix : ``str``
        Prefix of filenames of neighbour list
    scoring_method : ``pycentre.centretypes.ScoringMethods`` default: MIE
        A scoring method used for preparing convergance report from
        CENTRE output dataset scored against method input parametres.
    foretimators : ``str`` a rangestring or all
        estimators-index-set for which report is to be created.
    forbinschemes : ``str`` a rangestring or all
        binschemes-index-set for which report is to be created.
    forsteps : ``str`` a rangestring or all
        forsteps-index for which report is to be created, and coorlation
        graph is generated.
    writemiststeps : ``str`` a rangestring
        indices of steps for which MIST tree file will be written.
    gengraph : boolean default=False
        Generate a networkx.Graph and return if True or None if False
    threshold : ``double`` default=1.0e-8
        A real thresold value, pairs with MI  <= thresold will be considered
        zero and will not be included in graph-edge creation and MI scoring.
    outfileext : ``str``
        Extension of convergance report filename
    debug_state : ``boolean`` default False

    Returns
    -------
    networkx.Graph or None
        A networkx.Graph of MI values with edges define for MI > thresold
        for the final estimator in list, final binscheme and final step will
        be returned. ``If scoring_method != ScoringMethods.MIE else None``

    """
    graph = None
    s1_dict = None
    if isinstance(inputs, centretypes.CentreInputs):
        if scoring_method != centretypes.ScoringMethods.MIE:
            gengraph = True
        step_size = inputs.entropy.nframeseff / inputs.control.nsteps
        ent_contri_files = {}
        dir1 = inputs.control.outfilepath
        writemiststeps_rng = str2range(
            writemiststeps, last=inputs.control.nsteps)
        print("MIST WriteSteps: ", writemiststeps_rng)

        estimator_eff = []
        binsscemes_eff = []
        steps_eff = []
        if foretimators == 'all':
            estimator_eff = [(ii, ei)
                             for ii, ei in enumerate(inputs.estimators)]
        else:
            foretimators_list = str2range(
                foretimators, last=len(inputs.estimators))
            estimator_eff = [(ei, inputs.estimators[ei])
                             for ei in foretimators_list]

        if inputs.bats.pdfmethod == centretypes.PDFMethod.HISTOGRAM:
            if forbinschemes == 'all':
                binsscemes_eff = [(ii, ssi)
                                  for ii, ssi in enumerate(inputs.hist.binschemes)]
            else:
                forbinschemes_list = str2range(
                    forbinschemes, last=len(inputs.hist.binschemes))
                binsscemes_eff = [(ssi, inputs.hist.binschemes[ssi])
                                  for ssi in forbinschemes_list]
        else:
            binsscemes_eff = [(0, inputs.vmkde.nmaxconf)]

        if forsteps == 'all':
            steps_eff = [ii for ii in range(inputs.control.nsteps)]
        else:
            steps_list = str2range(
                forsteps, last=inputs.control.nsteps)
            steps_eff = [ssi for ssi in steps_list]
        print("STEPS: ", steps_eff)
        workset = inputs.entropy.workset

        ineigh = inputs.neighbors
        isubset = inputs.subset
        isubset.set_moltree(moltree)
        idpair2indexmap = neighbordofs.NeighborDOFsIds2IndexMap()
        ineigh.setIdpair2IndexMap(idpair2indexmap, init_idpairdict=True)

        if workset.has(centretypes.BATSet.B1D) or workset.has(centretypes.BATSet.BB2D):
            ent_contri_files['b'] = nc.Dataset(dir1 + '/entcontri_bnd-1d.nc')
            ids_list = ent_contri_files['b'].variables['id'][:]
            k_idx = 0
            for id in ids_list:
                iid = int(id)
                if iid in isubset.bnd_ids_:
                    isubset.bnd_ids_[iid] = k_idx
                else:
                    raise Exception("bond id ({0}) does not exist in SubsetDoFs".format(iid))
                isubset.bnd_idx_[k_idx]  = iid
                k_idx += 1
        if workset.has(centretypes.BATSet.A1D) or workset.has(centretypes.BATSet.AA2D):
            ent_contri_files['a'] = nc.Dataset(dir1 + '/entcontri_ang-1d.nc')
            ids_list = ent_contri_files['a'].variables['id'][:]
            k_idx = 0
            for id in ids_list:
                iid = int(id)
                if iid in isubset.ang_ids_:
                    isubset.ang_ids_[iid] = k_idx
                else:
                    raise Exception("angle id ({0}) does not exist in SubsetDoFs".format(iid))
                isubset.ang_idx_[k_idx]  = iid
                k_idx += 1
        if workset.has(centretypes.BATSet.D1D) or workset.has(centretypes.BATSet.DD2D):
            ent_contri_files['t'] = nc.Dataset(dir1 + '/entcontri_tor-1d.nc')
            ids_list = ent_contri_files['t'].variables['id'][:]
            k_idx = 0
            for id in ids_list:
                iid = int(id)
                if iid in isubset.tor_ids_:
                    isubset.tor_ids_[iid] = k_idx
                else:
                    raise Exception("torsion id ({0}) does not exist in SubsetDoFs".format(iid))
                isubset.tor_idx_[k_idx]  = iid
                k_idx += 1
        if workset.has(centretypes.BATSet.BB2D):
            ent_contri_files['bb'] = nc.Dataset(dir1 + '/entcontri_bnd-2d.nc')
            ids_list = ent_contri_files['bb'].variables['id'][:]
            k_idx = 0
            for ii in range(ids_list.shape[0]):
                id_pair = tuple(ids_list[ii])
                if ((id_pair[0] in ineigh.idpair2indexmap.bb_ids_) and
                    (id_pair[1] in ineigh.idpair2indexmap.bb_ids_[id_pair[0]])):
                    ineigh.idpair2indexmap.bb_ids_[id_pair[0]][id_pair[1]] = k_idx
                else:
                    raise Exception("bond-bond id pair ({0}, {1}) does not exist in neighbours".format(id_pair[0], id_pair[1]))
                #ineigh.bb_idx_[k_idx] = id_pair
                k_idx += 1
        if workset.has(centretypes.BATSet.AA2D):
            ent_contri_files['aa'] = nc.Dataset(dir1 + '/entcontri_ang-2d.nc')
            ids_list = ent_contri_files['aa'].variables['id'][:]
            k_idx = 0
            for ii in range(ids_list.shape[0]):
                id_pair = tuple(ids_list[ii])
                if ((id_pair[0] in ineigh.idpair2indexmap.aa_ids_) and
                    (id_pair[1] in ineigh.idpair2indexmap.aa_ids_[id_pair[0]])):
                    ineigh.idpair2indexmap.aa_ids_[id_pair[0]][id_pair[1]] = k_idx
                else:
                    raise Exception("angle-angle id pair ({0}, {1}) does not exist in neighbours".format(id_pair[0], id_pair[1]))
                # ineigh.aa_idx_[k_idx] = id_pair
                k_idx += 1
        if workset.has(centretypes.BATSet.DD2D):
            ent_contri_files['tt'] = nc.Dataset(dir1 + '/entcontri_tor-2d.nc')
            ids_list = ent_contri_files['tt'].variables['id'][:]
            k_idx = 0
            for ii in range(ids_list.shape[0]):
                id_pair = tuple(ids_list[ii])
                if ((id_pair[0] in ineigh.idpair2indexmap.tt_ids_) and
                    (id_pair[1] in ineigh.idpair2indexmap.tt_ids_[id_pair[0]])):
                    ineigh.idpair2indexmap.tt_ids_[id_pair[0]][id_pair[1]] = k_idx
                else:
                    raise Exception("torsion-torsion id pair ({0}, {1}) does not exist in neighbours".format(id_pair[0], id_pair[1]))
                # ineigh.tt_idx_[k_idx] = id_pair
                k_idx += 1
        if workset.has(centretypes.BATSet.B1D) and\
                workset.has(centretypes.BATSet.A1D) and workset.has(centretypes.BATSet.BA2D):
            ent_contri_files['ba'] = nc.Dataset(dir1 + '/entcontri_ba-2d.nc')
            ids_list = ent_contri_files['ba'].variables['id'][:]
            k_idx = 0
            for ii in range(ids_list.shape[0]):
                id_pair = tuple(ids_list[ii])
                if ((id_pair[0] in ineigh.idpair2indexmap.ba_ids_) and
                    (id_pair[1] in ineigh.idpair2indexmap.ba_ids_[id_pair[0]])):
                    ineigh.idpair2indexmap.ba_ids_[id_pair[0]][id_pair[1]] = k_idx
                else:
                    raise Exception("bond-angle id pair ({0}, {1}) does not exist in neighbours".format(id_pair[0], id_pair[1]))
                # ineigh.ba_idx_[k_idx] = id_pair
                k_idx += 1
        if workset.has(centretypes.BATSet.B1D) and\
                workset.has(centretypes.BATSet.D1D) and workset.has(centretypes.BATSet.BD2D):
            ent_contri_files['bt'] = nc.Dataset(dir1 + '/entcontri_bd-2d.nc')
            ids_list = ent_contri_files['bt'].variables['id'][:]
            k_idx = 0
            for ii in range(ids_list.shape[0]):
                id_pair = tuple(ids_list[ii])
                if ((id_pair[0] in ineigh.idpair2indexmap.bt_ids_) and
                    (id_pair[1] in ineigh.idpair2indexmap.bt_ids_[id_pair[0]])):
                    ineigh.idpair2indexmap.bt_ids_[id_pair[0]][id_pair[1]] = k_idx
                else:
                    raise Exception("bond-torsion id pair ({0}, {1}) does not exist in neighbours".format(id_pair[0], id_pair[1]))
                # ineigh.bt_idx_[k_idx] = id_pair
                k_idx += 1
        if workset.has(centretypes.BATSet.A1D) > 0 and\
                workset.has(centretypes.BATSet.D1D) > 0 and workset.has(centretypes.BATSet.AD2D):
            ent_contri_files['at'] = nc.Dataset(dir1 + '/entcontri_ad-2d.nc')
            ids_list = ent_contri_files['at'].variables['id'][:]
            k_idx = 0
            for ii in range(ids_list.shape[0]):
                id_pair = tuple(ids_list[ii])
                if ((id_pair[0] in ineigh.idpair2indexmap.at_ids_) and
                    (id_pair[1] in ineigh.idpair2indexmap.at_ids_[id_pair[0]])):
                    ineigh.idpair2indexmap.at_ids_[id_pair[0]][id_pair[1]] = k_idx
                else:
                    raise Exception("angle-torsion id pair ({0}, {1}) does not exist in neighbours".format(id_pair[0], id_pair[1]))
                # ineigh.at_idx_[k_idx] = id_pair
                k_idx += 1

        ineigh.idpair2indexmap.is_ready = True
        inputs.neighbors = ineigh
        inputs.subset = isubset

        obj_sub = deepcopy(inputs.subset)
        obj_sub.set_moltree(moltree)

        if len(cutoffs) == 0:
            # INFINITE_NEIGHBOR_CUTOFF represents infinite cut i.e. MIE/MIST
            cutoffs.append(INFINITE_NEIGHBOR_CUTOFF)

        for cut in cutoffs:
            neigh_current = None
            if cut != INFINITE_NEIGHBOR_CUTOFF:
                neigh_cut = neighbordofs.NeighborDOFs()
                neigh_cut.set_subset(obj_sub)
                neigh_cut.load(inputs.control.infilepath + os.sep +
                               '%s_%s.txt' % (neigh_fprefix, cut), workset)
                if debug_state:
                    print("Cutoff neigh on load: ", str(neigh_cut.get_typewise_pairs_counts()), flush=True)
                neigh_current = ineigh.intersection(neigh_cut)
                neigh_current.setIdpair2IndexMap(ineigh.idpair2indexmap, False)
                if debug_state:
                    print("Intersection neigh: ", str(neigh_current.get_typewise_pairs_counts()), flush=True)
            else:
                neigh_current = ineigh
                if debug_state:
                    print("Setting neigh_current=ineigh")
            for estm_i, estm in estimator_eff:
                for bi, bins in binsscemes_eff:
                    if cut != INFINITE_NEIGHBOR_CUTOFF:
                        fname = "{0}{1}{2}-{3}_convergence_b-{4}_n{5}.{6}".format(
                            inputs.control.outfilepath, os.sep, scoring_method.name, estm.name, bins, cut, outfileext)
                    else:
                        fname = "{0}{1}{2}-{3}_convergence_b-{4}.{5}".format(
                            inputs.control.outfilepath, os.sep, scoring_method.name, estm.name, bins, outfileext)
                    report = ConverganceReport(fname)
                    print("Working on file: " + fname, flush=True)
                    print("Number of pairs in neighborhoods:\n\t Calculated {} \n"
                          "\t Current {} pairs.".format(str(ineigh.get_typewise_pairs_counts()),
                                      str(neigh_current.get_typewise_pairs_counts())), flush=True)
                    for si in steps_eff:
                        si_row = report.get_row_odict()
                        si_row['TIME'] = (si + 1) * step_size
                        if cut != INFINITE_NEIGHBOR_CUTOFF:
                            mistfname = "{0}{1}Tree-{2}-{3}_b-{4}_s-{5}_n{6}.{7}".format(
                                inputs.control.outfilepath, os.sep, scoring_method.name, estm.name, bins, (si + 1), cut, outfileext)
                        else:
                            mistfname = "{0}{1}Tree-{2}-{3}_b-{4}_s-{5}.{6}".format(
                                inputs.control.outfilepath, os.sep, scoring_method.name, estm.name, bins, (si + 1), outfileext)
                        iswrtmist = si in writemiststeps_rng
                        s1_dict, graph = cal_conv_report_row(inputs,
                                                    ent_contri_files,
                                                    estm_i,
                                                    bi,
                                                    si,
                                                    neigh_current,
                                                    si_row,
                                                    threshold,
                                                    gengraph,
                                                    iswrtmist,
                                                    mistfname,
                                                    debug_state)
                        report.add_row(si_row)
                        print('.', end='', flush=True)
                        report.write()
                    print()

        if workset.has(centretypes.BATSet.B1D) or workset.has(centretypes.BATSet.BB2D):
            ent_contri_files['b'].close()
        if workset.has(centretypes.BATSet.A1D) or workset.has(centretypes.BATSet.AA2D):
            ent_contri_files['a'].close()
        if workset.has(centretypes.BATSet.D1D) or workset.has(centretypes.BATSet.DD2D):
            ent_contri_files['t'].close()
        if workset.has(centretypes.BATSet.BB2D):
            ent_contri_files['bb'].close()
        if workset.has(centretypes.BATSet.AA2D):
            ent_contri_files['aa'].close()
        if workset.has(centretypes.BATSet.DD2D):
            ent_contri_files['tt'].close()
        if workset.has(centretypes.BATSet.B1D) and\
                workset.has(centretypes.BATSet.A1D) and workset.has(centretypes.BATSet.BA2D):
            ent_contri_files['ba'].close()
        if workset.has(centretypes.BATSet.B1D) and\
                workset.has(centretypes.BATSet.D1D) and workset.has(centretypes.BATSet.BD2D):
            ent_contri_files['bt'].close()
        if workset.has(centretypes.BATSet.A1D) > 0 and\
                workset.has(centretypes.BATSet.D1D) > 0 and workset.has(centretypes.BATSet.AD2D):
            ent_contri_files['at'].close()

    else:
        print("ERROR:: Received: input type is %s" % type(inputs))
    return s1_dict, graph


class DOFGroups(object):
    """
    A Class for representing a mapping between the degrees-of-freedom of
    the molecule and the groups. where grouping can be done on one of the
    following bases:
    {
    'res' : group DOFs on basis of residue which they are member of.
    'doftype' : group DOFs on basis of DOF type, DOF can be one fo the
                 following types: Bond(B), angle(A), torsion(T) or pseudo(P).
    'torname' : Torsions have been named accoring to standard conventions,
                torsions not been given standard name have been given names.
                Here all the torsions of molecules are grouped by their
                names.
    'torids' :  Torsions are grouped by their ids.
    'doftype_and_res' : This grouping allows every residue to have three
                         B/A/T groups thus in total 3 x n_res groups are
                         created.
    }

    """
    def __init__(self):
        """
        Construct a object
        """
        self.name2id = collections.OrderedDict()
        self.id2name = collections.OrderedDict()
        self.dofid2gname = {}
        self.data = None
        self.frequency = None
        self.ready = False
        self.groupby = ""
        self.dims_scored = 0

    def freeze_schema(self):
        """
        This method is used for flagging a DOFGroup objects ready status.
        Because scoring of contributions of group has to have the groups
        defined in the first place. This flag is `False` until groups are
        created.
        """
        self.ready = True

    def score_groups_I2(self, graph, dims=1):
        """
        Score contribution of groups to I2.

        Parameters
        ----------
        graph : A networkx.Graph object returned from make_convergence_report
                This graph has DOFs as nodes and MI & -MI as weights for pair
                of its nodes, while these pairs of nodes come from the
                NeighborDOFs.

        dims : int choose {1, 2}
               dims = 1, if marginal MI contributions of the groups are to be
               calculated and reported.
               dims = 2, if group-group MI contributions are to be calculated

        Returns
        -------
        numpy.array : sum_mi (1-d if dims=1, 2-d if dims=2)
            Call to this method mutates the state of the object and returned
            data is also stored in `self.data` field of object.
            This method does return a numpy.array (1-d if dims=1, 2-d if dims=2)
            However mapping between indices of the array and the group name is
            maintained in `self.name2id` and `self.id2name` fields of the object.
        numpy.array : frequency (1-d if dims=1, 2-d if dims=2)
        """
        if dims==1:
            self.data = np.zeros(len(self.name2id))
            self.frequency = np.zeros(len(self.name2id), dtype=int)
            for ed in graph.edges(data=True):
                if ed[0] in self.dofid2gname and ed[1] in self.dofid2gname:
                    grp1, grp2 = self.dofid2gname[ed[0]], self.dofid2gname[ed[1]]
                    if grp1 == grp2:
                        self.data[self.name2id[grp1]] += ed[2]['mi']
                        self.frequency[self.name2id[grp1]] += 1
                    else:
                        self.data[self.name2id[grp1]] += ed[2]['mi'] / 2.0
                        self.data[self.name2id[grp2]] += ed[2]['mi'] / 2.0
                        self.frequency[self.name2id[grp1]] += 1
                        self.frequency[self.name2id[grp2]] += 1
            self.dims_scored = 1
        elif dims == 2:
            self.data = np.reshape(np.zeros(len(self.name2id)**2), (len(self.name2id), len(self.name2id)))
            self.frequency = np.reshape(np.zeros(len(self.name2id)**2, dtype=int), (len(self.name2id), len(self.name2id)))
            for ed in graph.edges(data=True):
                if ed[0] in self.dofid2gname and ed[1] in self.dofid2gname:
                    grp1, grp2 = self.dofid2gname[ed[0]], self.dofid2gname[ed[1]]
                    gid1, gid2 = self.name2id[grp1], self.name2id[grp2]

                    # Here 2D array maintains symmetry, and only tril, triu is represents
                    # all relevant info
                    if grp1 == grp2:
                        self.data[gid1, gid2] += ed[2]['mi']
                        self.frequency[gid1, gid2] += 1
                    else:
                        # Assumes gid1 > gid2, freq stored in transpose positions
                        grow_data, gcol_data = gid1, gid2
                        if gid1 < gid2:
                            grow_data, gcol_data = gid2, gid1
                        self.data[grow_data, gcol_data] += ed[2]['mi']
                        self.frequency[grow_data, gcol_data] += 1

            self.dims_scored = 2
        return self.data, self.frequency


    def score_groups_S1(self, s1dict):
        """
        Calculate the contributions to first order entropy S(1),
        according to defined groupings.

        Parameters
        ----------
        s1dict : `dict`, a ditionary of first order S(1) by individual
                 DOFs of the molecules where DOFid is Key and corresponding
                 value is its contribution to S(1).
                 A DOFid is of form `Xi` where X is a char from {B,A,T} &
                 i is the index of DF e.g if DOF is 15th Bond then it would
                 be 'B15'

        Returns
        -------
        numpy.array : an 1d numpy array, of groups contribution to S(1).
        numpy.array : an 1d numpy array, of groups frequencies.
        """
        self.data = np.zeros(len(self.name2id))
        self.frequency = np.zeros(len(self.name2id), dtype=int)
        for ek, ed in s1dict.items():
            if ek in self.dofid2gname:
                grp1 = self.dofid2gname[ek]
                self.data[self.name2id[grp1]] += ed
                self.frequency[self.name2id[grp1]] += 1
            elif (self.groupby == 'torname' or self.groupby == 'torids') and ek[0] == 'T':
                print("Torsion DOF not grouped ", ek[0], ek , ed)
            elif not (self.groupby == 'torname' or self.groupby == 'torids'):
                print("DOF not grouped ", ek , ed)
        self.dims_scored = 1
        return self.data, self.frequency


    def to_graph(self):
        """
        This method returns a networkx.Graph object, with groups as nodes
        and group-group pairs as edges, egde dict has two keys
        mi_sum & mi_inv where MI sum for the group-group DOFs and mi_inv is
        1.0/(1.0+mi_sum) : 1.0 is added in denomirator to avoid the singularity
        fo mi_sum==0.0, since MI is always a 0 or positive quantity range of
        mi_inv is [1, 0).
        """
        graph = None
        if self.dims_scored == 2:
            graph = nx.Graph()
            for i1, n1 in self.name2id.items():
                for i2, n2 in self.name2id.items():
                    graph.add_edge(n1, n2, mi_sum = self.data[i1][i2], mi_inv=1.0/float(1.0 + self.data[i1][i2]))
        else:
            print("It's possible only if groups are scored with dims=2.")
        return graph


    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.

def save_DOFGroups(dofgroups, dump_file):
    with open(dump_file, 'wb') as f:
        pickle.dump(dofgroups, f, pickle.HIGHEST_PROTOCOL)



def define_dofs_groupings(tree, groupby='res', moldefs=None):
    """
    Parameters
    ----------
    tree    : a BATtree object for the molecular system.
    groupby : `str`
                choose from
                {
                'res' : group DOFs on basis of residue which they are member of.
                'doftype' : group DOFs on basis of DOF type, DOF can be one fo the
                             following types: Bond(B), angle(A), torsion(T) or pseudo(P).
                'torname' : Torsions have been named accoring to standard conventions,
                            torsions not been given standard name have been given names.
                            Here all the torsions onf molecules are grouped by their
                            names.
                'torids' :  Torsions are grouped by their ids.
                'doftype_and_res' : This grouping allows every residue to have three
                                    B/A/T groups thus in total 3 x n_res groups are
                                    created.
                'doftype_and_mol' : This grouping allows every molecules to have three
                                    B/A/T groups thus in total 3 x n_mols groups are
                                    created.' while a dict of mol-names and a lsit of
                                    member residues is to be provided.
                }
    moldefs : `dict` only needed for ``groupby=='doftype_and_mol'``
              a dictionary with molecule-name as key and a list as value of member
              residues of the molecule.

    Returns
    -------
    groups : a DOFGroups object
    """
    groups = DOFGroups()
    grpidx = 0
    for i, nd in tree.nodes.items():
        if groupby == 'res':
            if nd.bnd_idx > 0:
                if nd.bnd_res not in groups.name2id:
                    groups.name2id[nd.bnd_res] = grpidx
                    groups.id2name[grpidx] = nd.bnd_res
                    grpidx += 1
                groups.dofid2gname['B{0}'.format(nd.bnd_idx-1)] = nd.bnd_res
            if nd.ang_idx > 0:
                if nd.ang_res not in groups.name2id:
                    groups.name2id[nd.ang_res] = grpidx
                    groups.id2name[grpidx] = nd.ang_res
                    grpidx += 1
                groups.dofid2gname['A{0}'.format(nd.ang_idx-1)] = nd.ang_res
            if nd.tor_idx > 0:
                if nd.tor_res not in groups.name2id:
                    groups.name2id[nd.tor_res] = grpidx
                    groups.id2name[grpidx] = nd.tor_res
                    grpidx += 1
                groups.dofid2gname['T{0}'.format(nd.tor_idx-1)] = nd.tor_res
        elif groupby == 'doftype':
            if nd.bnd_idx > 0:
                if 'B' not in groups.name2id:
                    groups.name2id['B'] = grpidx
                    groups.id2name[grpidx] = 'B'
                    grpidx += 1
                if nd.bnd_res > 0:
                    groups.dofid2gname['B{0}'.format(nd.bnd_idx-1)] = 'B'
                elif nd.bnd_res < 0:
                    if 'P' not in groups.name2id:
                        groups.name2id['P'] = grpidx
                        groups.id2name[grpidx] = 'P'
                        grpidx += 1
                    groups.dofid2gname['B{0}'.format(nd.bnd_idx-1)] = 'P'
            if nd.ang_idx > 0:
                if 'A' not in groups.name2id:
                    groups.name2id['A'] = grpidx
                    groups.id2name[grpidx] = 'A'
                    grpidx += 1
                if nd.ang_res > 0:
                    groups.dofid2gname['A{0}'.format(nd.ang_idx-1)] = 'A'
                elif nd.ang_res < 0:
                    if 'P' not in groups.name2id:
                        groups.name2id['P'] = grpidx
                        groups.id2name[grpidx] = 'P'
                        grpidx += 1
                    groups.dofid2gname['A{0}'.format(nd.ang_idx-1)] = 'P'
            if nd.tor_idx > 0:
                if 'T' not in groups.name2id:
                    groups.name2id['T'] = grpidx
                    groups.id2name[grpidx] = 'T'
                    grpidx += 1
                if nd.tor_res > 0:
                    groups.dofid2gname['T{0}'.format(nd.tor_idx-1)] = 'T'
                elif nd.tor_res < 0:
                    if 'P' not in groups.name2id:
                        groups.name2id['P'] = grpidx
                        groups.id2name[grpidx] = 'P'
                        grpidx += 1
                    groups.dofid2gname['T{0}'.format(nd.tor_idx-1)] = 'P'
        elif groupby == 'doftype_and_res':
            if nd.bnd_idx > 0:
                tmp1 = 'Br{0}'.format(nd.bnd_res)
                if tmp1 not in groups.name2id:
                    groups.name2id[tmp1] = grpidx
                    groups.id2name[grpidx] = tmp1
                    grpidx += 1
                groups.dofid2gname['B{0}'.format(nd.bnd_idx-1)] = tmp1
            if nd.ang_idx > 0:
                tmp1 = 'Ar{0}'.format(nd.ang_res)
                if tmp1 not in groups.name2id:
                    groups.name2id[tmp1] = grpidx
                    groups.id2name[grpidx] = tmp1
                    grpidx += 1
                groups.dofid2gname['A{0}'.format(nd.ang_idx-1)] = tmp1
            if nd.tor_idx > 0:
                tmp1 = 'Tr{0}'.format(nd.tor_res)
                if tmp1 not in groups.name2id:
                    groups.name2id[tmp1] = grpidx
                    groups.id2name[grpidx] = tmp1
                    grpidx += 1
                groups.dofid2gname['T{0}'.format(nd.tor_idx-1)] = tmp1
        elif groupby == 'doftype_and_mol':
            if nd.bnd_idx > 0:
                molname = ''
                for mn in moldefs.keys():
                    if nd.bnd_res in moldefs[mn]:
                        molname = mn
                        break
                if molname:
                    tmp1 = 'Bm{0}'.format(molname)
                    if tmp1 not in groups.name2id:
                        groups.name2id[tmp1] = grpidx
                        groups.id2name[grpidx] = tmp1
                        grpidx += 1
                    groups.dofid2gname['B{0}'.format(nd.bnd_idx-1)] = tmp1
            if nd.ang_idx > 0:
                molname = ''
                for mn in moldefs.keys():
                    if nd.ang_res in moldefs[mn]:
                        molname = mn
                        break
                if molname:
                    tmp1 = 'Am{0}'.format(molname)
                    if tmp1 not in groups.name2id:
                        groups.name2id[tmp1] = grpidx
                        groups.id2name[grpidx] = tmp1
                        grpidx += 1
                    groups.dofid2gname['A{0}'.format(nd.ang_idx-1)] = tmp1
            if nd.tor_idx > 0:
                molname = ''
                for mn in moldefs.keys():
                    if nd.tor_res in moldefs[mn]:
                        molname = mn
                        break
                if molname:
                    tmp1 = 'Tm{0}'.format(molname)
                    if tmp1 not in groups.name2id:
                        groups.name2id[tmp1] = grpidx
                        groups.id2name[grpidx] = tmp1
                        grpidx += 1
                    groups.dofid2gname['T{0}'.format(nd.tor_idx-1)] = tmp1
        elif groupby == 'torname':
            if nd.tor_idx > 0:
                if nd.name not in groups.name2id:
                    groups.name2id[nd.name] = grpidx
                    groups.id2name[grpidx] = nd.name
                    grpidx += 1
                groups.dofid2gname['T{0}'.format(nd.tor_idx-1)] = nd.name
        elif groupby == 'torids':
            if nd.tor_idx > 0:
                if nd.tor_idx not in groups.name2id:
                    groups.name2id[nd.tor_idx] = grpidx
                    groups.id2name[grpidx] = nd.tor_idx
                    grpidx += 1
                groups.dofid2gname['T{0}'.format(nd.tor_idx-1)] = nd.tor_idx
    groups.groupby = groupby
    groups.freeze_schema()
    return groups


def read_csv_as_dict(filename, header=True, skipTopNlines=0):
    data = {}
    with open(filename, "r") as f:
        reader = csv.reader(f)
        heads = []
        for ri, row in enumerate(reader):
            if ri < skipTopNlines:
                pass
            elif ri == skipTopNlines:
                if header:
                    for k in row:
                        data[k.strip()] = []
                        heads.append(k.strip())
                else:
                    for i, v in enumerate(row):
                        data['V'+str(i+1)] = [v.strip()]
                        heads.append('V'+str(i+1))
            else:
                for i, c in enumerate(row):
                    data[heads[i]].append(float(c.strip()))
    return(data)

def get_poorly_poupulated_percent(data, nbins, ndata, thresold_perc=1.0):
    """
    Get the percentage of bins been under populated.
    """
    min_count = thresold_perc * ndata / (100*nbins)
    min_count = 2 if min_count < 2 else min_count
    n_almost_blanks = len(data[data <= min_count])
    res = round(n_almost_blanks * 100.0 / nbins,1)
    return res

class HistEntropySummary(object):
    def __init__(self, fname):
        self.fname = fname
        self.threshold = 1.0
        self.nframe = 0
        self.bin_schemes = []
        self.poorlypopulated = {}
        self.d2I2 = {}

    def run_analysis(self, fentpref, fentext='.csv'):
        ds = nc.Dataset(self.fname)
        self.bin_schemes = getattr(ds,'binSchemes')
        ent = {}
        for bi in sorted(self.bin_schemes):
            ent[bi] = read_csv_as_dict(fentpref+ str(bi) + fentext)

        for k in sorted(ent.keys()):
            self.d2I2[k] = (np.array(ent[k]['S2'][-1]) - np.array(ent[k]['S1'][-1]))
            nfrm = int(ent[k]['TIME'][-1])
            if self.nframe == 0:
                self.nframe = nfrm
            elif self.nframe != nfrm:
                raise Exception("Number of frames for all steps must be same found("+str(self.nframe) + ", " + str(nfrm) + ")")

        v1 = ds.variables['freq']
        for d in range(v1.shape[0]):
            offset_b = 0
            for i in self.bin_schemes:
                nbins = int(i) * int(i)
                vsubset = v1[d,-1, offset_b:offset_b+nbins]
                # print(offset_b, nbins, vsubset.shape)
                prc = get_poorly_poupulated_percent(vsubset, nbins, self.nframe, self.threshold)
                offset_b += nbins
                if i in self.poorlypopulated:
                    self.poorlypopulated[i].append(prc)
                else:
                    self.poorlypopulated[i] = [prc]
            if d % 1000 == 0:
                print(".", end='', flush=True)


    def make_plots(self):
        print("This function is not implemented yet!")
        pass

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.

