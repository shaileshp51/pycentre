# Installation
**Supported platforms**
 - Linux 
 - macOS
 
**Supported python versions**
- python >= 3.6

**Requires**
- cython >= 0.18
- h5py>=2.6
- netCDF4 >=1.2.7
- numpy>=1.10
- pandas>=0.19
- networkx>=1.11
- pytraj >=1.0

## Install required packages

**Recommended: using anaconda or miniconda **

> The simplest way to install miniconda is as below, open terminal and run below commands.

`wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
`bash Miniconda3-latest-Linux-x86_64.sh`

accept license and follow instructions.

> Once anaconda/miniconda is successfully installed. Close and reopen terminal.

> Its better to create a dedicated virtual environment to keep things clean. Now let's create a virtual environment say `venv_pycentre` as below.

`conda create -n venv_pycentre python=3.7`

> Now let's install all required packages except `pytraj`.

`conda install -n venv_pycentre h5py netcdf4 numpy pandas networkx`

> Now let's install `pytraj`

`conda install -n venv_pycentre -c ambermd pytraj`

> First activate the virtual environment for pycentre, now we are ready to install pycentre

`conda activate venv_pycentre`

> Now get pycentre source code from git

`git clone https://github.com/shaileshp51/pycentre.git`
`cd pycentre`

`# Linux`
`python ./setup.py install`

`# macOS:`
`python setup.py install`

## Install when required packages already installed

** from source code (for Linux and macOS) **

`git clone https://github.com/shaileshp51/pycentre.git`

`cd pycentre`

`# Linux`
`python ./setup.py install`

`# macOS:`
`python setup.py install`
> Note: here python referes to python3, which may be accessible as python if its a conda virtual environment. If the python is system python3 you may have to use sudo to grant write permission for installation.

# Usage

Now we are ready to start converting cartesian trajectories into BAT trajectory using `cart2bat.py`. All the options supported by `cart2bat.py` can be found in help message, which can be seen as:
`cart2bat.py --help`
