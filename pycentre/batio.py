# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
#from memory_profiler import profile

try:
    from netCDF4 import Dataset
except ImportError as e:
    raise Exception("Error in importing netCDF4 \n{0}".format(e))
#from memory_profiler import profile


class ncbat:
    """A netcdf trajectory containing internal coordinates (bond,angle,dihedral) with
    a set of methods to create / read / write and get details.

    Attributes:
        filename: int
            A string representing the name of file on which operations will be performed.
        n_atom: int
            A positive integer for storing number of atoms in system to be stored.
        n_bond: int
            A positive integer for storing number of bonds in system to be stored.
        n_angle: int 
            A positive integer for storing number of angles in system to be stored.
        n_dihedral: int
            A positive integer for storing number of dihedrals in system to be stored.
        root_indices: list[int] len==3
            A list or tuple with three positive integer elements representing indices of atoms
                      used to construct the internal coordinate tree, index based on *atom_start_index*.
        pseudo_bonds: list[int]/tuple[int]
            A list/tuple with two positive integer elements of a list of such lists or tuples,
                      providing pair of atom indices involved in pseudo bond used to construct internal
                      coordinate tree.
        atom_start_index: int
            An integer (0 or 1) representing starting index of atoms in system
    """

    def __init__(self, filename, n_atom, n_bond, n_angle, n_dihedral, root_indices, pseudo_bonds=None,
                 atom_start_index=1):
        """Return an Internal Coordinate NetCDF Trajectory manipulation object
        """
        v_pseudo = 'N'
        n_pseudo = 1
        pseudos_list = [[-1, -1]]
        if not (type(n_atom) == int and n_atom >= 0):
            raise Exception(
                "n_atom must be a positive integer provided {0}:".format(n_atom))
        if not (type(n_bond) == int and n_bond >= 0):
            raise Exception(
                "n_bond must be a positive integer provided {0}:".format(n_bond))
        if not (type(n_angle) == int and n_angle >= 0):
            raise Exception(
                "n_angle must be a positive integer provided {0}:".format(n_angle))
        if not (type(n_dihedral) == int and n_dihedral >= 0):
            raise Exception(
                "n_dihedral must be a positive integer provided {0}:".format(n_dihedral))

        if type(root_indices) in [list, tuple]:
            if not len(root_indices) == 3 and all(type(e) == int for e in root_indices):
                raise Exception(
                    "Exactly 3-atom indices are accepeted as root_indices")
        else:
            raise Exception(
                "pseudo_bonds can be a `2-tuple` of `int` or a `list` thereof")
        if pseudo_bonds is not None:
            if not (type(pseudo_bonds) in [list, tuple]):
                raise Exception(
                    "pseudo_bonds can be a `2-tuple` of `int` or a `list` thereof")
            else:
                isPairs = True
                if type(pseudo_bonds) is list:
                    isPairs = all(type(item) in [tuple, list] and len(
                        item) == 2 for item in pseudo_bonds)
                    # if not isPairs:
                    #    isPairs = len(pseudo_bonds) == 2 and all(type(item) is int for item in pseudo_bonds)
                else:
                    isPairs = isPairs and len(self.pseudo_bonds) == 2
                if not isPairs:
                    raise Exception(
                        "pseudo_bonds can be a `2-tuple` of `int` or a `list` thereof")
                else:
                    v_pseudo = 'Y'
                    if type(pseudo_bonds) is list:
                        n_pseudo = len(pseudo_bonds)
                        pseudos_list = [list(e) for e in pseudo_bonds]
                    else:
                        n_pseudo = 1
                        pseudos_list = list(pseudo_bonds)
        else:
            v_pseudo = 'N'
        self._filename = filename
        self._n_atom = n_atom
        self._n_bond = n_bond
        self._n_angle = n_angle
        self._n_dihedral = n_dihedral
        self._root_indices = root_indices
        self._pseudo_bonds = pseudo_bonds
        self._atom_start_index = atom_start_index
        self._application = "CENTRE"
        self._program = "Cart2BAT"
        self._program_version = "1.0"
        self._convention = "CENTRE"
        self._convention_version = "1.0"
        self._v_pseudo = v_pseudo
        self._n_pseudo = n_pseudo
        self._pseudo_list = pseudos_list

    def __str__(self):
        strVal = '''
        [ filename: {0},
          n_atom: {1},
          n_bond: {2},
          n_angle: {3},
          n_dihedral: {4},
          root_indices: {5},
          pseudo_bonds: {6},
          atom_start_index: {7},
          application: {8},
          program: {9},
          program_version: {10},
          convention: {11},
          convention_version: {12},
          has_pseudo: {13},
          n_pseudo: {14},
          pseudos_list: {15}
        ]'''.format((self._filename, self._n_atom, self._n_bond, self._n_angle, self._n_dihedral,
                     str(self._root_indices), self._pseudo_bonds,
                     self._atom_start_index, self._application, self._program,
                     self._program_version, self._convention, self._convention_version,
                     self._v_pseudo, self._n_pseudo, str(self._pseudo_list)
                     ))
        return(strVal)

    def create_dataset(self, title=""):
        try:
            # NETCDF3_64BIT_OFFSET
            rootgrp = Dataset(self._filename, "w", format="NETCDF4")
            try:
                rootgrp.title = title
                rootgrp.application = self._application
                rootgrp.program = self._program
                rootgrp.programVersion = self._program_version
                rootgrp.convention = self._convention
                rootgrp.conventionVersion = self._convention_version
                rootgrp.atomStartIndex = str(self._atom_start_index)
                rrt = self._root_indices
                rootgrp.rootIndices = "%d %d %d" % (rrt[0], rrt[1], rrt[2])
                rootgrp.hasPseudo = self._v_pseudo

                rootgrp.createDimension("tuple2", 2)
                rootgrp.createDimension("pseudos", self._n_pseudo)

                pseudo_bonds = rootgrp.createVariable(
                    "pseudoBonds", "i4", ("pseudos", "tuple2",))
                pseudo_bonds[:, :] = self._pseudo_list
                rootgrp.createDimension("frame", None)

                rootgrp.createDimension("bond", self._n_bond)
                rootgrp.createDimension("angle", self._n_angle)
                rootgrp.createDimension("dihedral", self._n_dihedral)

                times = rootgrp.createVariable("time", "f4", ("frame",))
                times.units = "picosecond"
                bonds = rootgrp.createVariable(
                    "bond", "f4", ("frame", "bond",), chunksizes=(4096, 4,))
                bonds.units = "angstrom"
                angles = rootgrp.createVariable(
                    "angle", "f4", ("frame", "angle",), chunksizes=(4096, 4,))
                angles.units = "degree"
                dihedrals = rootgrp.createVariable(
                    "dihedral", "f4", ("frame", "dihedral",), chunksizes=(4096, 4,))
                dihedrals.units = "degree"
            except IOError as e:
                raise IOError(
                    "Error in creating dimension/variables/attributes in netCDF file: {0}\n{1}".format(self._filename, e))
            finally:
                rootgrp.close()
                return True
        except IOError as e:
            raise IOError(
                "Unable to create netcdf file: {0}\n{1}".format(self._filename, e))

    def read_frame_times(self, indices=None):
        """Return values of time variable for given indices or all if indices is `None`
        """
        try:
            with Dataset(self._filename, "r") as d:
                try:
                    if indices is not None and type(indices) == tuple:
                        if 0 < len(indices) < 4:
                            _indices = [i for i in range(*indices)]
                            data = d.variables["time"][_indices]
                        else:
                            raise Exception(
                                "Invalid slice indices {0}".format(indices))
                    elif indices is not None and type(indices) == list:
                        data = d.variables["time"][indices]
                    elif indices is not None and type(indices) == int:
                        data = d.variables["time"][indices]
                    elif indices is None:
                        data = d.variables["time"][:]
                    else:
                        raise Exception(
                            "*indices* can only be a list or positive integers, a 1/2/3-tuple of positive integers or None")
                except:
                    raise Exception("Error encountered in reading times `%s` from file `%s`".format(
                        indices, self._filename))

                return(data)
        except:
            Exception(
                "Error encountered in opening file `%s` for reading".format(self._filename))
            return None

    def read_frame_coordinates(self, indices=None):
        """Return values of time variable for given indices or all if indices is `None`
        """
        try:
            with Dataset(self._filename, "r") as d:
                try:
                    if (indices is not None) and type(indices) == tuple:
                        if 0 < len(indices) < 4:
                            _indices = [i for i in range(*indices)]
                            print(_indices)
                            bnd = d.variables["bond"][_indices, :]
                            ang = d.variables["angle"][_indices, :]
                            dih = d.variables["dihedral"][_indices, :]
                        else:
                            raise Exception(
                                "Invalid slice indices {0}".format(indices))
                    elif (indices is not None) and type(indices) == list:
                        bnd = d.variables["bond"][indices, :]
                        ang = d.variables["angle"][indices, :]
                        dih = d.variables["dihedral"][indices, :]
                    elif indices is None:
                        bnd = d.variables["bond"][:, :]
                        ang = d.variables["angle"][:, :]
                        dih = d.variables["dihedral"][:, :]
                    else:
                        raise Exception(
                            "*indices* can only be a list or positive integers, a 1/2/3-tuple of positive integers or None")
                except:
                    raise Exception("Error encountered in reading coordinates `%s` from file `%s`".format(
                        indices, self._filename))
                return(bnd, ang, dih)
        except:
            raise Exception(
                "Error encountered in opening file `%s` for reading".format(self._filename))
            return (None, None, None)

    def read_frame_vars(self, frame_variable, frame_indices=None, var_indices=None):
        """Return values of time variable for given indices or all if indices is `None`
        """
        if frame_variable not in ["bond", "angle", "dihedral"]:
            print("{0} is not a frame vairable".format(frame_variable))
            return None
        try:
            with Dataset(self._filename, "r") as d:
                try:
                    if (frame_indices is not None) and type(frame_indices) == tuple:
                        if 0 < len(frame_indices) < 4:
                            _frame_indices = [i for i in range(*frame_indices)]
                        else:
                            frame_indices = None
                    elif (frame_indices is not None) and type(frame_indices) == list:
                        _frame_indices = frame_indices[:]
                    else:
                        frame_indices = None

                    if (var_indices is not None) and type(var_indices) == tuple:
                        if 0 < len(var_indices) < 4:
                            _var_indices = [i for i in range(*var_indices)]
                        else:
                            var_indices = None
                    elif(var_indices is not None) and type(var_indices) == list:
                        _var_indices = var_indices[:]
                    else:
                        var_indices = None

                    if frame_indices is not None:
                        if var_indices is not None:
                            var_data = d.variables[frame_variable][_frame_indices,
                                                                   :_var_indices]
                        else:
                            var_data = d.variables[frame_variable][_frame_indices, :]
                    else:
                        if var_indices is not None:
                            var_data = d.variables[frame_variable][:,
                                                                   _var_indices]
                        else:
                            var_data = d.variables[frame_variable][:, :]
                except:
                    raise Exception("Error encountered in reading variable `{0}` from file `{1}`".format(
                        frame_variable, self._filename))
                return(var_data)
        except:
            raise Exception(
                "Error encountered in opening file `{0}` for reading".format(self._filename))
            return None

    #@profile
    def append_frames(self, frame_times, frames_bonds, frames_angles, frames_dihedrals):
        try:
            n_frm_old = -1
            n_frm_new = -1
            with Dataset(self._filename, 'a') as d:
                try:
                    bonds = d.variables["bond"]
                    angles = d.variables["angle"]
                    dihedrals = d.variables["dihedral"]
                    times = d.variables["time"]
                    n_frames = d.dimensions["frame"].size

                    times[n_frames:] = frame_times
                    bonds[n_frames:, :] = frames_bonds
                    angles[n_frames:, :] = frames_angles
                    dihedrals[n_frames:, :] = frames_dihedrals
                    n_frm_old = n_frames
                    n_frm_new = n_frames + len(frame_times)
                    d.sync()
                except IOError as e:
                    raise IOError(
                        "Error in writing to file {0}\n{1}".format(self._filename, e))
        except IOError as e:
            raise IOError("Error in opening file `{0}` for writing, error: {1}".format(
                self._filename, e))
        return(n_frm_old, n_frm_new)

    def info(self):
        try:
            with Dataset(self._filename, 'r') as d:
                try:
                    print("NetCDF Dimensions of " + self._filename)
                    print("+" * 100 + "\n")
                    print(d.dimensions)
                    print("=" * 100 + "\n")
                    print("NetCDF Attributes of " + self._filename)
                    print("+" * 100 + "\n")
                    for attrib in d.ncattrs():
                        print("{0} : {1}".format(attrib, getattr(d, attrib)))
                    print("=" * 100 + "\n")
                    print("NetCDF Variables of " + self._filename)
                    print("+" * 100 + "\n")
                    print(d.variables)
                    print("=" * 100 + "\n")
                except (Exception, e):
                    raise Exception(
                        "Error in getting infomation from file `{0}` {1}".format(self._filename, e))
        except (Exception, e):
            raise Exception(
                "Error in opening file `{0}` {1}".format(self._filename, e))


"""
def main():
    traj = trjNetCdfBAT("test.nc", 33, 32,31,30, [1,2,3], pseudo_bonds=[(4, 8), (12, 16), (21, 23)])
    print(str(traj))
    if traj.create_dataset("First Trajectory"):
        print("Successfully created trajectory file.")
    print("\n\nPrint netCDF structure information of created file: {0}".format(traj._filename))
    traj.netcdf_info()
    #traj.write_to_netcdf('test.nc', tms, np.transpose(dist), np.transpose(angs), np.transpose(dihs))


if __name__ == "__main__":
    main()
"""
