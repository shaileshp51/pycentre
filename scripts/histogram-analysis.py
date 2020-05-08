#!/usr/bin/env python

import os
import re
import sys

import csv
import math
import pickle
import textwrap
import argparse

import numpy as np
import netCDF4 as nc

import matplotlib.pyplot as plt
from scipy import stats
from pycentre import *
from pycentre.analysis import *

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


parser = CustomArgumentParser(fromfile_prefix_chars='@', add_help=False)
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument("-hfile",
                    "--hist-file",
                    help="histogram file containing bin frequencies.",
                    required=True)
requiredNamed.add_argument("-dfile",
                    "--dump-file",
                    help="filename to be written with anaylysis data",
                    required=True)
requiredNamed.add_argument("-fentpref",
                    "--entropy-report-file-prefix",
                    help="entropy-report-file-prefix",
                    required=True)
parser.add_argument("-fxt",
                    "--fentext",
                    default='.csv',
                    help="file-extension of entropy-report-file [default='.csv']")
parser.add_argument("-t",
                    "--threshold",
                    default=1.0,
                    help="percent value for threshold below which bin is flagged poorly-populated [defaut=1.0]")


child_parser = CustomArgumentParser(fromfile_prefix_chars='@', parents=[parser],
                                    prog='PROG',
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    description=textwrap.dedent('''\
         Please do consider possibility given below as well!
         ----------------------------------------------------------------------------
             You can keep these command-line-argument(s) and their values in a file
             say (cmd_args.txt), each option in seperate-line, you can also keep
             comment lines starting with character '#' in the file, Then involke
             program to read it from file using:
             PROG @cmd_args.txt
         ****************************************************************************
         '''))
        
def main():
    args = child_parser.parse_args()
    
    histsummary = HistEntropySummary(args.hist_file)
    histsummary.threshold = args.threshold
    print(histsummary.threshold)
    histsummary.run_analysis(args.entropy_report_file_prefix, args.fentext)
    with open(args.dump_file, 'wb') as f:
        pickle.dump(histsummary, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()




