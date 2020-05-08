import os

class Struct(object):
    def __init__(self, adict):
        """
        Convert a dictionary to a class

        Parameter
        ---------
        adict : ``dict``
            Input Dictionary Object

        """
        self.__dict__.update(adict)
        for k, v in adict.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(v)


def get_object(adict):
    """
    Convert a dictionary to a class

    Parameter
    ---------
    adict : ``dict``
        Input Dictionary Object

    Return
    ------
        class:Struct

    """
    return Struct(adict)


def readlines_delim(f, delimiter):
    """
    A generator for yielding lines from a text-file handle open for reading.

    Parameter
    ---------
    f : filehandle
        A handle to a text-file opened for reading
    delimiter : str
        Read text from inputfile handle one-line at a time with line ending
        marked by it.

    Yields
    ------
        str
        A string containing a line from the file first line if called first on 
        the file or next line of previous call.

    """
    buf = ""
    while True:
        while delimiter in buf:
            pos = buf.index(delimiter)
            yield buf[:pos]
            buf = buf[pos + len(delimiter):]
        chunk = f.read(4096)
        if not chunk:
            yield buf
            break
        buf += chunk


def bins_poorstats(data, nbins, ndata, threshold_percent=1.0):
    """
    Get Percentage of total bins which are poorly populated

    Parameters
    ----------
    data : numpy.ndarray[int, ndim=1]
        A numpy array holding observed frequencies of bins used for histogram
    nbins : int
        number of bins
    ndata : int
        Number of sample points to be histogrammed.
    threshold_percent : float 
        A float value in an open interval(0.0, 100.0) bins with occupancy(%)
        below which will be labelled blank(poorly-populated). Here, it is 
        assumed that sample-data is from uniform distribution, and thershold
        is percent-thresold of expected occupancy.

        e.g. If data has 1000000 sample points, hisstogrammed with 200 bins, 
        then with 1.0% thershold value. bins having less-than 
        (1.0/100.0) * (1000000 / 50.0) i.e. 50 samples will be considered
        poorly populated, if it comes less than 2 then minimum observed freq is 
        considered 2.

    Returns
    -------
    float 
        A value in an open interval(0.0, 100.0) representing percentage of 
        total bins labeled blank(poorly-populated)

    """
    min_count = threshold_percent * ndata / (100 * nbins)
    min_count = 2 if min_count < 2 else min_count
    n_almost_blanks = len(data[data <= min_count])
    res = round(n_almost_blanks * 100.0 / nbins, 1)
    return res


def stripquotes(inpstring, both=False, quotechar='"'):
    """
    Return an unquoted str of supplied str

    Parameters
    ----------
    inpstring : str
        Input string
    both : boolean, optional default False
        Strip both single and double quotes from input if True, otherwise
        strip single-quote or double-quote as specified by ``param quotechar``
    quotechar : str 
        A string from {"'", '"'} for striping single-quote or double-quote
        respectively.

    Returns
    -------
    str
        Stripped string.

    """
    if not both:
        if inpstring.startswith(quotechar) and inpstring.endswith(quotechar):
            return inpstring[1:-1]
    else:
        if inpstring.startswith('"') and inpstring.endswith('"'):
            return inpstring[1:-1]
        elif inpstring.startswith("'") and inpstring.endswith("'"):
            return inpstring[1:-1]


def get_absfile(path_string, start_dir, mode):
    """
    Get absolute path of the file
    
    Parameters
    ----------
    path_string : str
        a string representing the relative or absolute path of the file.
    start_dir : str
        if path_string is a relative path, then relative path will be converted to
        absolute path with `start_dir` as the prefix of the `start_dir`
    mode : str choose {'r', 'w', 'x'}
        if mode is r/x then the quiried file path has be existing, if it is w
        then the directory of the absolute path has to be existing.
        
    Returns
    -------
    str
        a string of the absolute path is returned.
    """
    result = True
    msg = ''
    absfile = ''
    val = os.path.isabs(path_string)
    if val:
        result = os.path.exists(path_string)
        absfile = path_string
    else:
        pth = os.path.abspath(os.path.join(os.sep, start_dir, path_string))
        if mode == 'r':
            result = os.path.exists(pth)
            if not result:
                msg = "`%s` file doesn't exists" % (pth)
            else:
                absfile = pth
        elif mode == 'w':
            result = os.path.isdir(os.path.split(pth)[0])
            if not result:
                msg = '`%s` is not a directory' % (os.path.split(pth)[0])
            else:
                absfile = pth
        elif mode == 'x':
            result = os.path.isdir(os.path.split(pth)[0])
            if result:
                result = result and os.path.exists(pth)
                if result:
                    msg = "`%s` file exists can't be created" % (pth)
                    result = not result
                else:
                    absfile = pth
            else:
                msg = '`%s` is not a directory' % (os.path.split(pth)[0])
        if not result:
            raise Exception(msg)
    return absfile


def is_directory(dir_string):
    """
    Is it a directory?

    Returns
    -------
    boolean
        True if so otherwise False
    """
    return os.path.isdir(dir_string)


def is_validfile(argmnt, path_string, start_dir, mode):
    """
    Is it a valid file with mode
    
    """
    result = True
    msg = ''
    val = os.path.isabs(path_string)
    if val:
        result = os.path.exists(path_string)
    else:
        pth = os.path.abspath(os.path.join(os.sep, start_dir, path_string))
        if mode == 'r':
            result = os.path.exists(pth)
            if not result:
                msg = "validating (%s): `%s` file doesn't exists" % (
                    argmnt, pth)
        elif mode == 'w':
            result = os.path.isdir(os.path.split(pth)[0])
            if not result:
                msg = 'validating (%s): `%s` is not a directory' % (
                    argmnt, os.path.split(pth)[0])
        elif mode == 'x':
            result = os.path.isdir(os.path.split(pth)[0])
            if result:
                result = result and os.path.exists(pth)
                if result:
                    msg = "validating (%s): `%s` file exists can't be created" % (
                        argmnt, pth)
                    result = not result
            else:
                msg = 'validating (%s): `%s` is not a directory' % (
                    argmnt, os.path.split(pth)[0])
        if not result:
            raise Exception(msg)
    return result


# TODO: implementation is to be tested
def list_to_ranges(list_of_ints, range_sep=',', slice_sep=':'):
    """
    Get a ranges-str for input list

    Parameters
    ----------
    list_of_ints : list[int]
        A list of integers to codify as a ranges-str
    range_sep : str default ','
        A range seperator string
    slice_sep : str default ':'
        A slice seperator string, range with only number `x` will be treated 
        as singleton integer memeber (not a 0..x range)
        
    Returns
    -------
    str
        A ranges_string, with ranges delimited by ',' and slice char ':'

    """
    t = [(i - 1, list_of_ints[i] - list_of_ints[i - 1]) for i in range(1, len(list_of_ints))]
    i = 0
    e = t[i]
    i += 1
    res = []
    while i < len(t):
        is_range = False
        if t[i][1] - e[1] == 0:
            while i < len(t) and t[i][1] - e[1] == 0:
                i += 1
                is_range = True
            if is_range:
                f = t[i - 1]
                r1 = (list_of_ints[e[0]], list_of_ints[f[0] + 1], f[1]
                      ) if f[1] != 1 else (list_of_ints[e[0]], list_of_ints[f[0] + 1])
                res.append(r1)
            else:
                res.extend([list_of_ints[e[0]], list_of_ints[e[0]]])
            i += 1
        else:
            res.extend([list_of_ints[e[0]]])

        if i < len(t):
            e = t[i]
            i += 1
            if i == len(t):
                res.extend([list_of_ints[e[0]], list_of_ints[e[0]]])
    rs1 = range_sep.join([slice_sep.join(str(n) for n in elm) if isinstance(
        elm, tuple) else str(elm) for elm in res])
    return rs1


def ranges_to_list(range_str, range_sep=',', slice_sep=':'):
    """
    Get a list of integers represented by input ranges-str

    Parameters
    ----------
    range_str : str
        A ranges_string, with ranges delimited by ',' and slice char ':'
    range_sep : str default ','
        A range seperator string
    slice_sep : str default ':'
        A slice seperator string, range with only number `x` will be treated 
        as singleton integer memeber (not a 0..x range)

    Returns
    -------
    list[int]
        A list of integer-members of specified ranges-string

    """
    rngs = range_str.split(range_sep)
    res = []
    for r in rngs:
        slice_e = r.split(slice_sep)
        if len(slice_e) > 1:
            rng = [int(i.strip()) for i in slice_e]
            res.extend(range(*rng))
        elif len(slice_e) == 1:
            s = int(slice_e[0].strip())
            res.append(s)
    return(res)


def is_equal_2d(ndarray1, ndarray2, showdiff=False, tolerance=0.000001):
    """
    Whether supplied 2d numpy arrays are similar within the tolerance limit?
    
    Parameters
    ----------
    ndarray1 :  numpy.ndarray[float64, ndim=2]
        First array supplied for equality with `ndarray2`.
    ndarray2 :  numpy.ndarray[float64, ndim=2]
        Second array supplied for equality angainst `ndarray1`.
    showdiff : bool (default: False)
        If two array corresponding elements differ by morre than `tolerance`
        then stop further comparison and report position of mismatch.
    tolerance : float64 (default: 0.000001)
        A threshold value which can be tolerated for similarity of two 
        corresponding elements of the supplied arrays.
        
    Returns
    -------
    bool, str, list[str] if showdiff==True
    bool otherwise
        A thruth value of equality query of arguments, difference accepted
        within tolerance.
        Cause why two arguments dont match.
        A list of string showing first position and values of entries which
        do not match.
    """
    result = True
    cause = ''
    diff_loc_val = []
    if type(ndarray1) is not type(ndarray2):
        result = False
        cause = 'Dataset type mismatch: {0} and {1}' % (
            type(ndarray1), type(ndarray2))
    else:
        if ndarray1.shape != ndarray2.shape:
            result = False
            cause = 'dataset shape mismatch: {0} and {1}' % (
                ndarray1.shape, ndarray2.shape)
        else:
            for i in range(ndarray1.shape[0]):
                for j in range(ndarray1.shape[1]):
                    d = ndarray1[i][j] - ndarray2[i][j]
                    if d > tolerance:
                        diff_loc_val.append(
                            (i, j, ndarray1[i][j], ndarray2[i][j], d))
                        result = False
                        break
                else:
                    continue  # executed if the loop ended normally (no break)
                break  # exec
    if showdiff:
        return (result, cause, diff_loc_val)
    else:
        return result


def is_equal_1d(ndarray1, ndarray2, showdiff=False, tolerance=0.000001):
    """
    Whether supplied 1d numpy arrays are similar within the tolerance limit?
    
    Parameters
    ----------
    ndarray1 :  numpy.ndarray[float64, ndim=1]
        First array supplied for equality with `ndarray2`.
    ndarray2 :  numpy.ndarray[float64, ndim=1]
        Second array supplied for equality angainst `ndarray1`.
    showdiff : bool (default: False)
        If two array corresponding elements differ by morre than `tolerance`
        then stop further comparison and report position of mismatch.
    tolerance : float64 (default: 0.000001)
        A threshold value which can be tolerated for similarity of two 
        corresponding elements of the supplied arrays.
        
    Returns
    -------
    bool, str, list[str] if showdiff==True
    bool otherwise
        A thruth value of equality query of arguments, difference accepted
        within tolerance.
        Cause why two arguments dont match.
        A list of string showing first position and values of entries which
        do not match.
    """
    result = True
    cause = ''
    diff_loc_val = []
    if type(ndarray1) is not type(ndarray2):
        result = False
        cause = 'Dataset type mismatch: {0} and {1}' % (
            type(ndarray1), type(ndarray2))
    else:
        if ndarray1.shape != ndarray1.shape:
            result = False
            cause = 'dataset shape mismatch: {0} and {1}' % (
                ndarray1.shape, ndarray2.shape)
        else:
            for i in range(ndarray1.shape[0]):
                d = ndarray1[i] - ndarray2[i]
                if d > tolerance:
                    diff_loc_val.append((i, d))
                    result = False
                    break
    if showdiff:
        return (result, cause, diff_loc_val)
    else:
        return result
