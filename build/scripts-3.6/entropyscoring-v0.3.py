#!/home/shailesh/anaconda3/envs/venv_pycentre/bin/python

from memory_profiler import profile
import sys
import os
import re
import textwrap
import argparse

try:
    from pycentre import centretypes as pyctypes
    from pycentre import mmsystem as pycmms
    from pycentre import analysis as pycanalysis
except ImportError:
    import pycentre.centretypes as pyctypes
    import pycentre.mmsystem as pycmms
    import pycentre.analysis as pycanalysis


def is_directory(dir_string):
    val = os.path.isdir(dir_string)
    if not val:
        raise argparse.ArgumentTypeError(
            "`%s` is not a valid directory" % dir_string)
    return dir_string


class CentreOutputAnalyzer:
    def __init__(self, input):
        self.centreinput = input
        self.ncexpectfiles = {}
        self.ncoutputfiles = []
        self.convgexpectfiles = []
        self.convgfiles = []
        self.mistreefiles = []

    def checkintigrity(self):
        entc_files = []
        if self.centreinput.entropy.workset.has(pyctypes.BATSet.B1D) or\
                self.centreinput.entropy.workset.has(pyctypes.BATSet.BB2D):
            entc_files.append("entcontri_bnd-1d.nc")
        if self.centreinput.entropy.workset.has(pyctypes.BATSet.BB2D):
            entc_files.append("entcontri_bnd-2d.nc")
        if self.centreinput.entropy.workset.has(pyctypes.BATSet.A1D) or\
                self.centreinput.entropy.workset.has(pyctypes.BATSet.AA2D):
            entc_files.append("entcontri_ang-1d.nc")
        if self.centreinput.entropy.workset.has(pyctypes.BATSet.AA2D):
            entc_files.append("entcontri_ang-2d.nc")
        if self.centreinput.entropy.workset.has(pyctypes.BATSet.D1D) or\
                self.centreinput.entropy.workset.has(pyctypes.BATSet.DD2D):
            entc_files.append("entcontri_tor-1d.nc")
        if self.centreinput.entropy.workset.has(pyctypes.BATSet.DD2D):
            entc_files.append("entcontri_tor-2d.nc")
        if self.centreinput.entropy.workset.has(pyctypes.BATSet.BA2D):
            entc_files.append("entcontri_ba-2d.nc")
        if self.centreinput.entropy.workset.has(pyctypes.BATSet.BD2D):
            entc_files.append("entcontri_bd-2d.nc")
        if self.centreinput.entropy.workset.has(pyctypes.BATSet.AD2D):
            entc_files.append("entcontri_ad-2d.nc")

        if entc_files:
            self.ncexpectfiles['entropy'] = entc_files

        hist_files = []
        if self.centreinput.hist.writefreq or self.centreinput.vmkde.writefreq:
            writeset = pyctypes.BATSet.NOTHING
            if self.centreinput.bats.pdfmethod == pyctypes.PDFMethod.HISTOGRAM:
                writeset = self.centreinput.hist.writefreq
            elif self.centreinput.bats.pdfmethod == pyctypes.PDFMethod.vonMisesKDE:
                writeset = self.centreinput.vmkde.writefreq
            if writeset.has(pyctypes.BATSet.B1D) or writeset.has(pyctypes.BATSet.BB2D):
                hist_files.append("hist_bnd-1d.nc")
            if writeset.has(pyctypes.BATSet.BB2D):
                hist_files.append("hist_bnd-2d.nc")
            if writeset.has(pyctypes.BATSet.A1D) or writeset.has(pyctypes.BATSet.AA2D):
                hist_files.append("hist_ang-1d.nc")
            if writeset.has(pyctypes.BATSet.AA2D):
                hist_files.append("hist_ang-2d.nc")
            if writeset.has(pyctypes.BATSet.D1D) or writeset.has(pyctypes.BATSet.DD2D):
                hist_files.append("hist_tor-1d.nc")
            if writeset.has(pyctypes.BATSet.DD2D):
                hist_files.append("hist_tor-2d.nc")
            if writeset.has(pyctypes.BATSet.BA2D):
                hist_files.append("hist_ba-2d.nc")
            if writeset.has(pyctypes.BATSet.BD2D):
                hist_files.append("hist_bd-2d.nc")
            if writeset.has(pyctypes.BATSet.AD2D):
                hist_files.append("hist_ad-2d.nc")

        if hist_files:
            self.ncexpectfiles['histogram'] = hist_files

        for f in self.ncexpectfiles['entropy']:
            if not os.path.isfile(f):
                raise Exception("FileNotFound: " + f)
        for f in self.ncexpectfiles['histogram']:
            if not os.path.isfile(f):
                raise Exception("FileNotFound: " + f)




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
                        raise Exception("Unrecognised range string: " + tmp)
            except Exception:
                raise Exception("Unrecognised range string: " + tmp)
        setattr(namespace, self.dest, atoms)


parser = CustomArgumentParser(fromfile_prefix_chars='@', add_help=False)
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument("-i",
                           "--input",
                           help="input file used for centre run.",
                           required=True)
requiredNamed.add_argument("-t",
                           "--tree-file",
                           help="filename for writing molecule-tree.",
                           required=True)
parser.add_argument("-m",
                    "--score-method",
                    default=pyctypes.ScoringMethods.MIE,
                    help="Entropy method for scoring. (default: 'MIE')")
parser.add_argument("-e",
                    "--estimators",
                    default="all",
                    help="estimators `rangestring` e.g. '0:4:1'  for entropy scoring. (default: 'all')")
parser.add_argument("-b",
                    "--bin-schemes",
                    default="all",
                    help="bin-schemes `rangestring` for entropy scoring. (default: 'all')")
parser.add_argument("-s",
                    "--score-steps",
                    default="all",
                    help="scoring for steps `rangestring` for  entropy. (default: 'all')")
parser.add_argument("-w",
                    "--write-mis-tree-steps",
                    default="9:last:10",
                    help="rangestring for writing mis tree. (default: '9:last:10')")
parser.add_argument("-o",
                    "--out-file",
                    default=sys.stdout,
                    help="filename for writing run output. (default: sys.stdout)")

child_parser = CustomArgumentParser(fromfile_prefix_chars='@', parents=[parser],
                                    prog='entropyscoring.py',
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    description=textwrap.dedent('''\
         Please do consider possibility given below as well!
         ----------------------------------------------------------------------------
             You can keep these command-line-argument(s) and their values in a file
             say (cmd_args.txt). Keeping each option in seperate-line, you can also
             comment lines starting with character '#' in the file, Then involke
             program to read it from file using:
             entropyscoring.py @cmd_args.txt
         ****************************************************************************
         '''))
subsetgrp = child_parser.add_argument_group('NeighborDOFS command group')
subsetgrp.add_argument("-cut", "--cutoff",
                       help="distance cutoff for neighbor consideration",
                       type=int, action="store", default=[], nargs='*')
subsetgrp.add_argument("-nneigh",
                       "--no-neighbor",
                       help="do not use neighbors for scoring, it a flag supplying it means True (default: False)",
                       action="store_true")
subsetgrp.add_argument("-nfp", "--neigh-fprefix",
                       help="filename prefix for neighbor used for scoring",
                       default="neigh")
subsetgrp.add_argument("-rfe", "--report-file-extension",
                       help="extension of report file extension of scoring",
                       default="txt")

# @profile
def main():
    args = child_parser.parse_args()
    if args.score_method == "MIST":
        args.score_method = pyctypes.ScoringMethods.MIST
    inputs = pyctypes.CentreInputs(args.input)
    print(inputs)

    moltree = pycmms.MMTree()
    moltree.load(inputs.control.infilepath + os.sep + args.tree_file)
    inputs.subset.set_moltree(moltree)
    cutoffs = [int(i) for i in args.cutoff]
    if args.no_neighbor:
        print("Note: --no-neighbor option supplied, so all cutoffs will be ignored!")
        cutoffs = []
    print(cutoffs)
    pycanalysis.make_conv_report(inputs, moltree, cutoffs, args.neigh_fprefix, scoring_method=args.score_method,
                                 foretimators=args.estimators, forbinschemes=args.bin_schemes,
                                 forsteps=args.score_steps, writemiststeps=args.write_mis_tree_steps,
                                 outfileext=args.report_file_extension, debug_state=True)


if __name__ == "__main__":
    main()
