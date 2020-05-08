from enum import Enum
from enum import IntEnum
from enum import IntFlag
#from configparser import ConfigParser

#import os
#
#try:
#    from pycentre import utils_ as utls
#    from pycentre import subneigh
#    from pycentre import subsetdofs
#    from pycentre import neighbordofs
#
#except ImportError:
#    try:
#        from . import utils_ as utls
#        from . import subneigh
#        import subsetdofs
#        import neighbordofs
#    except ImportError as err:
#        raise ("Error import: " + str(err))


# Coordinate System
# BAT: Bond Angle Torsion


class CoordSys(Enum):
    CARTESIAN = 0
    BAT = 1


class BAT_t(Enum):
    NONE = 0
    BOND = 1
    ANGLE = 2
    DIHEDRAL = 3


class BATSet(IntFlag):
    def has(self, b):
        return bool(self.value & b.value)

    NOTHING = 0      # None
    B1D = 1          # Bonds 1D
    A1D = 2          # Angles 1D
    D1D = 4          # Torsions 1D
    X1D = 7          # Bonds, Angles, Torsions 1D
    BB2D = 8         # Bonds 2D
    AA2D = 16        # Angles 2D
    DD2D = 32        # Torsions 2D
    XX2D = 63        # Bonds, Angles, Torsions 2D
    BA2D = 64        # Bonds-Angles 2D
    BD2D = 128       # Bonds-Torsions 2D
    AD2D = 256       # Angles-Torsions 2D
    XY2D = 511       # (Bonds, Angles, Torsions)-1D/2D,
    #                  Bonds-Angles 2D, Bonds-Torsions 2D, Angles-Torsions 2D


# ALL: Entire data avaliable in input data files will be used for entropy analysis
# SUBSET: A subset of entire data avaliable in input data files will be
# used for entropy analysis
class Entropy4Data(Enum):
    ALL = 0
    SUBSET = 1


class PDFMethod(Enum):
    HISTOGRAM = 0
    vonMisesKDE = 1


# MIE: Mutual Information Expansion
# AMIE: Approximate Mutual Information Expansion
# MIST: Mutual Information Spanning Tree
# MLA: Maximum Local Approximation
# CC_MLA: Correlation Corrected MLA
class ScoringMethods(Enum):
    MIE = 1     # Mutual Information Expansion
    AMIE = 2    # Approximate MIE, enabled when entropy.usesubset=True
    MIST = 3,   # Maximum Information Spanning Tree


class EntropyEstimators(Enum):
    ML = 0,  # ML: Maximum Likelihood Estimator
    MM = 1  # MM: Miller Maddow Estimator
    CS = 2  # CS: Chao Shen
    JS = 3  # James and Stein Shrinkage Estimator


#def stripquotes(inpstring, quotechar='"', both=False):
#    ostr = inpstring[:]
#    if not both:
#        if inpstring.startswith(quotechar) and inpstring.endswith(quotechar):
#            ostr = ostr[1:-1]
#    else:
#        if inpstring.startswith('"') and inpstring.endswith('"'):
#            return ostr[1:-1]
#        elif inpstring.startswith("'") and inpstring.endswith("'"):
#            ostr = ostr[1:-1]
#    return ostr
#
#
#class CentreInputs:
#    class Control:
#        def __init__(self):
#            self.nsteps = 40
#            self.discretize = True
#            self.calcentropy = True
#            self.genconvgdata = True
#            self.dscrinfofreq = 50
#            self.entcinfofreq = 500
#            self.outfilepath = None
#            self.infilepath = None
#            self.infofile = None
#
#        def __str__(self):
#            indent = " " * 4
#            lst = ["{0}{1:<20s} = {2}".format(indent, k, str(
#                v)) for k, v in self.__dict__.items() if v is not None]
#            return os.linesep.join(lst)
#
#    class Entropy:
#        def __init__(self):
#            self.nframeseff = 0
#            self.startframe = 0
#            self.strideframe = 1
#            self.nframes = 0
#            self.nfiles = 0
#            self.workset = BATSet.NOTHING
#            self.shuffleblocks = False
#            self.usesubset = False
#            self.useneighbor = False
#            self.jacobian = True
#            self.shuffleblocktimes = 1
#            self.subsetfile = None
#            self.neighborfile = None
#
#        def __str__(self):
#            indent = " " * 4
#            lst = ["{0}{1:<20s} = {2}".format(indent, k, str(
#                v)) for k, v in self.__dict__.items() if v is not None]
#            return os.linesep.join(lst)
#
#    class Discretization:
#        def __init__(self):
#            self.nbond = 0
#            self.nangle = 0
#            self.ndihed = 0
#            self.shuffleframes = False
#            self.optimizedih = True
#            self.shuffledofs = False
#            self.randseed = 4000
#            self.pdfmethod = PDFMethod.HISTOGRAM
#            self.fnames = []
#            self.nframes = []
#
#        def __str__(self):
#            indent = " " * 4
#            lst = ["{0}{1:<20s} = {2}".format(
#                indent, k, str(v)) for k, v in self.__dict__.items() if v is not None]
#            return os.linesep.join(lst)
#
#    def _batset_str2val(self, strvals):
#        reslt = BATSet.NOTHING
#        val = BATSet.NOTHING
#
#        for iv, strval in enumerate(strvals.replace(',', ' ').split()):
#            if strval == "NONE":
#                val = BATSet.NOTHING  # None
#                break
#            elif strval == "B1D":
#                val = BATSet.B1D  # Bonds1D
#            elif strval == "A1D":
#                val = BATSet.A1D  # Angles1D
#            elif strval == "D1D":
#                val = BATSet.D1D  # Torsions 1D
#            elif strval == "1D":
#                val = BATSet.X1D  # Bonds, Angles, Torsions 1D
#            elif strval == "BB2D":
#                val = BATSet.BB2D  # Bonds 2D
#            elif strval == "AA2D":
#                val = BATSet.AA2D  # Angles 2D
#            elif strval == "DD2D":
#                val = BATSet.DD2D  # Torsions 2D
#            elif strval == "XX2D":
#                val = BATSet.XX2D  # Bonds, Angles, Torsions 2D
#            elif strval == "BA2D":
#                val = BATSet.BA2D  # Bonds-Angles 2D
#            elif strval == "BD2D":
#                val = BATSet.BD2D  # Bonds-Torsions 2D
#            elif strval == "AD2D":
#                val = BATSet.AD2D  # Angles-Torsions 2D
#            elif strval == "2D":
#                # Bonds, Angles, Torsions 2D, Bonds-Angles 2D, Bonds-Torsions 2D,
#                # Angles-Torsions 2D
#                val = BATSet.XY2D
#
#            if iv == 0:
#                reslt = val
#            else:
#                reslt = reslt | val
#
#        return reslt
#
#    class vMisesKDE:
#        def __init__(self):
#            self.writefreq = False
#            self.writeset = BATSet.NOTHING
#            self.writestepstart = 0
#            self.writestepstride = 1
#            self.nmaxconf = 5
#            self.kappa = 1.0
#            self.sdonstep = 5
#            self.sdoiterations = 1000
#            self.sdoconvlimit = 0.0001
#
#        def __str__(self):
#            indent = " " * 4
#            lst = ["{0}{1:<20s} = {2}".format(
#                indent, k, str(v)) for k, v in self.__dict__.items() if v is not None]
#            return os.linesep.join(lst)
#
#    class Histogram:
#        def __init__(self):
#            self.writefreq = False
#            self.writeset = BATSet.NOTHING
#            self.writestepstart = 0
#            self.writestepstride = 1
#            self.referencenbins = 30
#            self.binschemes = []
#
#        def __str__(self):
#            indent = " " * 4
#            lst = ["{0}{1:<20s} = {2}".format(
#                indent, k, str(v)) for k, v in self.__dict__.items() if v is not None]
#            return os.linesep.join(lst)
#
#    def __init__(self, filename):
#        self.filename = filename
#        self.control = self.Control()
#        self.scoringmethod = ScoringMethods.MIST
#        self.estimators = []
#        self.entropy4data = Entropy4Data.ALL
#        self.bats = self.Discretization()
#        self.hist = self.Histogram()
#        self.vmkde = self.vMisesKDE()
#        self.entropy = self.Entropy()
#        self.subset = None
#        self.neighbors = None
#
#        cfg = ConfigParser(inline_comment_prefixes=(';', '#'))
#        cfg.read(filename)
#        if cfg:
#            for sec in cfg.sections():
#                if sec == 'control':
#                    if cfg.has_option(sec, 'infilepath'):
#                        self.control.infilepath = stripquotes(
#                            cfg.get(sec, 'infilepath'), both=True)
#                    if cfg.has_option(sec, 'outfilepath'):
#                        self.control.outfilepath = stripquotes(
#                            cfg.get(sec, 'outfilepath'), both=True)
#                    if cfg.has_option(sec, 'infofile'):
#                        self.control.infofile = stripquotes(
#                            cfg.get(sec, 'infofile'), both=True)
#                    if cfg.has_option(sec, 'discretize'):
#                        # print(cfg.get(sec, 'discretize'))
#                        self.control.discretize = cfg.getboolean(
#                            sec, 'discretize')
#                    if cfg.has_option(sec, 'calcentropy'):
#                        self.control.calcentropy = cfg.getboolean(
#                            sec, 'calcentropy')
#                    if cfg.has_option(sec, 'genconvgdata'):
#                        self.control.genconvgdata = cfg.getboolean(
#                            sec, 'genconvgdata')
#                    if cfg.has_option(sec, 'nsteps'):
#                        self.control.nsteps = cfg.getint(sec, 'nsteps')
#                    if cfg.has_option(sec, 'dscrinfofreq'):
#                        self.control.dscrinfofreq = cfg.getint(
#                            sec, 'dscrinfofreq')
#                    if cfg.has_option(sec, 'entcinfofreq'):
#                        self.control.entcinfofreq = cfg.getint(
#                            sec, 'entcinfofreq')
#
#                if sec == 'discretization':
#                    if cfg.has_option(sec, 'fname'):
#                        self.bats.fnames.append(stripquotes(
#                            cfg.get(sec, 'fname'), both=True))
#                    if cfg.has_option(sec, 'pdfmethod'):
#                        val = stripquotes(cfg.get(sec, 'pdfmethod'), both=True)
#                        if val.lower() == 'vonmiseskde':
#                            self.bats.pdfmethod = PDFMethod.vonMisesKDE
#                        elif val.lower() == 'histogram':
#                            self.bats.pdfmethod = PDFMethod.HISTOGRAM
#                    if cfg.has_option(sec, 'shuffleframes'):
#                        self.bats.shuffleframes = cfg.getboolean(
#                            sec, 'shuffleframes')
#                    if cfg.has_option(sec, 'shuffledofs'):
#                        self.bats.shuffledofs = cfg.getboolean(
#                            sec, 'shuffledofs')
#                    if cfg.has_option(sec, 'optimizedih'):
#                        self.bats.optimizedih = cfg.getboolean(
#                            sec, 'optimizedih')
#                    if cfg.has_option(sec, 'nbond'):
#                        self.bats.nbond = cfg.getint(sec, 'nbond')
#                    if cfg.has_option(sec, 'nangle'):
#                        self.bats.nangle = cfg.getint(sec, 'nangle')
#                    if cfg.has_option(sec, 'ndihed'):
#                        self.bats.ndihed = cfg.getint(sec, 'ndihed')
#                    if cfg.has_option(sec, 'nframe'):
#                        self.bats.nframes.append(cfg.getint(sec, 'nframe'))
#                    if cfg.has_option(sec, 'randseed'):
#                        self.bats.randseed = cfg.getint(sec, 'randseed')
#
#                if sec == 'histogram':
#                    if cfg.has_option(sec, 'writefreq'):
#                        self.hist.writefreq = cfg.getboolean(
#                            sec, 'writefreq')
#                    if cfg.has_option(sec, 'writeset'):
#                        self.hist.writeset = self._batset_str2val(cfg.get(
#                            sec, 'writeset'))
#                    if cfg.has_option(sec, 'writestepstart'):
#                        self.hist.writestepstart = cfg.getint(
#                            sec, 'writestepstart')
#                    if cfg.has_option(sec, 'writestepstride'):
#                        self.hist.writestepstride = cfg.getint(
#                            sec, 'writestepstride')
#                    if cfg.has_option(sec, 'referencenbins'):
#                        self.hist.referencenbins = cfg.getint(
#                            sec, 'referencenbins')
#                    if cfg.has_option(sec, 'nbins'):
#                        for b in cfg.get(sec, 'nbins').replace(',', ' ').split():
#                            self.hist.binschemes.append(int(b.strip()))
#
#                if sec == 'vonmiseskde':
#                    if cfg.has_option(sec, 'writefreq'):
#                        self.vmkde.writefreq = cfg.getboolean(
#                            sec, 'writefreq')
#                    if cfg.has_option(sec, 'writeset'):
#                        self.vmkde.writeset = self._batset_str2val(cfg.get(
#                            sec, 'writeset'))
#                    if cfg.has_option(sec, 'writestepstart'):
#                        self.vmkde.writestepstart = cfg.getint(
#                            sec, 'writestepstart')
#                    if cfg.has_option(sec, 'writestepstride'):
#                        self.vmkde.writestepstride = cfg.getint(
#                            sec, 'writestepstride')
#                    if cfg.has_option(sec, 'nmaxconf'):
#                        self.vmkde.nmaxconf = cfg.getint(sec, 'nmaxconf')
#                    if cfg.has_option(sec, 'sdosteps'):
#                        self.vmkde.sdosteps = cfg.getint(sec, 'sdosteps')
#                    if cfg.has_option(sec, 'sdoiterations'):
#                        self.vmkde.sdoiterations = cfg.getint(
#                            sec, 'sdoiterations')
#                    if cfg.has_option(sec, 'kappa'):
#                        self.vmkde.kappa = cfg.getfloat(sec, 'kappa')
#                    if cfg.has_option(sec, 'sdoconvlimit'):
#                        self.vmkde.sdoconvlimit = cfg.getfloat(
#                            sec, 'sdoconvlimit')
#
#                if sec == 'entropy':
#                    if cfg.has_option(sec, 'subsetfile'):
#                        self.entropy.subsetfile = stripquotes(
#                            cfg.get(sec, 'subsetfile'), both=True)
#                    if cfg.has_option(sec, 'neighborfile'):
#                        self.entropy.neighborfile = stripquotes(
#                            cfg.get(sec, 'neighborfile'), both=True)
#                    if cfg.has_option(sec, 'scoringmethod'):
#                        val = stripquotes(
#                            cfg.get(sec, 'scoringmethod'), both=True)
#                        if val == "MIE":
#                            self.scoringmethod = ScoringMethods.MIE
#                        elif val == "MIST":
#                            self.scoringmethod = ScoringMethods.MIST
#                    if cfg.has_option(sec, 'estimator'):
#                        for val in cfg.get(sec, 'estimator').replace(',', ' ').split():
#                            val = val.upper()
#                            if val == "ML":
#                                self.estimators.append(EntropyEstimators.ML)
#                            elif val == "MM":
#                                self.estimators.append(EntropyEstimators.MM)
#                            elif val == "CS":
#                                self.estimators.append(EntropyEstimators.CS)
#                            elif val == "JS":
#                                self.estimators.append(EntropyEstimators.JS)
#                    if cfg.has_option(sec, 'workset'):
#                        self.entropy.workset = self._batset_str2val(cfg.get(
#                            sec, 'workset'))
#                    if cfg.has_option(sec, 'shuffleblocks'):
#                        self.entropy.shuffleblocks = cfg.getboolean(
#                            sec, 'shuffleblocks')
#                    if cfg.has_option(sec, 'usesubset'):
#                        self.entropy.usesubset = cfg.getboolean(
#                            sec, 'usesubset')
#                    if cfg.has_option(sec, 'useneighbor'):
#                        self.entropy.useneighbor = cfg.getboolean(
#                            sec, 'useneighbor')
#                    if cfg.has_option(sec, 'jacobian'):
#                        self.entropy.jacobian = cfg.getboolean(
#                            sec, 'jacobian')
#                    if cfg.has_option(sec, 'shuffleblocktimes'):
#                        self.entropy.shuffleblocktimes = cfg.getint(
#                            sec, 'shuffleblocktimes')
#                    if cfg.has_option(sec, 'startframe'):
#                        self.entropy.startframe = cfg.getint(
#                            sec, 'startframe')
#                    if cfg.has_option(sec, 'strideframe'):
#                        self.entropy.strideframe = cfg.getint(
#                            sec, 'strideframe')
#                    if cfg.has_option(sec, 'nframe'):
#                        self.entropy.nframe = cfg.getint(
#                            sec, 'nframe')
#
#        numfrm = 0
#        for i in self.bats.nframes:
#            numfrm += i
#
#        if self.entropy.nframes > numfrm or self.entropy.nframes <= 0:
#            self.entropy.nframes = numfrm
#        self.entropy.nfiles = len(self.bats.fnames)
#        nframes_eff = 0
#        if self.entropy.nframes > 0 and (self.entropy.startframe > 0 or self.entropy.strideframe > 1):
#            nframes_eff = (self.entropy.nframes - self.entropy.startframe +
#                           self.entropy.strideframe - 1) // self.entropy.strideframe
#        else:
#            nframes_eff = self.entropy.nframes
#
#        self.entropy.nframeseff = nframes_eff
#        if self.entropy.usesubset:
#            abssub = utls.get_absfile(
#                self.entropy.subsetfile, self.control.infilepath, 'r')
#            print(abssub)
#            self.subset = subsetdofs.SubsetDOFs()
#            self.subset.load(abssub, self.entropy.workset)
#            print(self.subset)
#            self.entropy4data = Entropy4Data.SUBSET
#            if self.entropy.useneighbor:
#                absngh = utls.get_absfile(
#                    self.entropy.neighborfile, self.control.infilepath, 'r')
#                self.neighbors = neighbordofs.NeighborDOFs()
#                self.neighbors.set_subset(self.subset)
#                self.neighbors.load(absngh, self.entropy.workset)
#            else:
#                self.neighbors = neighbordofs.NeighborDOFs(self.subset, self.entropy.workset)
#
#        else:
#            self.subset = subsetdofs.SubsetDOFs(
#                n_bnd=self.bats.nbond, n_ang=self.bats.nangle, n_tor=self.bats.ndihed, self.entropy.workset)
#            self.neighbors = neighbordofs.NeighborDOFs(self.subset, self.entropy.workset)
#
#        if self.control.nsteps == 0:
#            self.control.nsteps = 1
#
#        if self.hist.writefreq and self.hist.writestepstart == 255:
#            self.hist.writestepstart = self.control.nsteps - 1
#
#        if self.bats.pdfmethod == PDFMethod.vonMisesKDE:
#            self.hist.referencenbins = self.vmkde.nmaxconf
#            self.hist.binschemes.append(self.vmkde.nmaxconf)
#
#        if len(self.estimators) == 0:
#            self.estimators.append(EntropyEstimators.ML)
#
#    def __str__(self):
#        lst = []
#        lst.append(" [control]")
#        lst.append(str(self.control))
#        lst.append('')
#        lst.append(" [discretization]")
#        lst.append(str(self.bats))
#        lst.append('')
#        if self.bats.pdfmethod == PDFMethod.HISTOGRAM:
#            lst.append(" [histogram]")
#            lst.append(str(self.hist))
#            lst.append('')
#        elif self.bats.pdfmethod == PDFMethod.vonMisesKDE:
#            lst.append(" [vonmiseskde]")
#            lst.append(str(self.vmkde))
#            lst.append('')
#        lst.append(" [entropy]")
#        lst.append(str(self.entropy))
#        indent = " " * 4
#        lst.append("{0}{1:<20s} = {2}".format(indent, "estimators",
#                                              ", ".join([es.name for es in self.estimators])))
#        lst.append("{0}{1:<20s} = {2}".format(indent, "scoringmethods",
#                                              self.scoringmethod.name))
#
#        return os.linesep.join(lst)
