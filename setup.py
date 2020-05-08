from setuptools import setup
#from distutils.core import setup
from setuptools.extension import Extension

from Cython.Build import cythonize

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension("pycentre.core", [ "pycentre/core.pyx" ]),
        Extension("pycentre.subsetdofs", [ "pycentre/subsetdofs.pyx" ]),
        Extension("pycentre.neighbordofs", [ "pycentre/neighbordofs.pyx" ]),
        Extension("pycentre.analysis", [ "pycentre/analysis.pyx" ]),
    ]
    for e in ext_modules:
        e.cython_directives = {'language_level': "3"}
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("pycentre.core", [ "pycentre/core.c" ]),
        Extension("pycentre.subsetdofs", [ "pycentre/subsetdofs.c" ]),
        Extension("pycentre.neighbordofs", [ "pycentre/neighbordofs.c" ]),
        Extension("pycentre.analysis", [ "pycentre/analysis.c" ]),
    ]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(
    name = "pycentre",
    packages = ["pycentre"],
    version = "0.1.0",
    package_data={'pycentre': ['data/*.lib']},
    description = "Cartesian to BAT coordinate system MD trajectory conversion",
    author = "Shailesh Kumar Panday",
    author_email = "shaileshp51@gmail.com",
    url = "http://ccbb.jnu.ac.in/",
    download_url = "http://ccbb.jnu.ac.in",
    keywords = ["trajectory", "md", "bat", "configurational entropy"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Development Status :: 1 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Researcheres, Computational Chemists, Computational Scientists",
        "License :: OSI Approved :: GNU General Public License version 3 or higher(GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Configurational entropy :: Molecular Dynamics :: Libraries",
    ],
    scripts=['scripts/cart2bat.py', 'scripts/entropyscoring-v0.3.py', 'scripts/histogram-analysis.py', 'scripts/neighborbuilder.py'],
    setup_requires=['Cython >= 0.18'],
    install_requires=[
        'pytraj>=1.0',
        'netCDF4>=1.2.7',
        'h5py>=2.6',
        'numpy>=1.10',
        'pandas>=0.19',
        'networkx>=1.11'
    ],
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    #ext_modules = cythonize("cart2intdev/core.pyx"),
    long_description = """\
Python Package for processing Molecular Dynamics Trajectories and Post processing CENRTE outputs.
-------------------------------------------------------------------------------------------------

Converts
 - NAMD/CHARMM/AMBER/GROMACS/ACEMD produced, DCD/NETCDF/mdcrd trajectories to produce BAT trajectory in netcdf fprmat
 - Creates molecular tree represnting the input system
 - Writes neighbor-list for combination of {bond,angle,torsion} taken two at a time


This version requires Python 3 or later.
"""
)
