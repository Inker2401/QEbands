"""
Read and parse various Quantum ESPRESSO input files
that are not currently supported by ASE.

Author: V Ravindran 31/03/2025
"""
import re
import xml.etree.ElementTree as ET

import numpy as np
from ase.units import Bohr


def read_pwscf_header(outfile: str) -> dict:
    """Read the contents of the PWSCF output file.

    The main reason to call this routine is if the XML file
    is not available to obtain otherwise missing information
    from the post-processing file.

    Parameters
    ----------
    outfile : str
        main output from PWSCF

    Returns
    -------
    dict
        various quantities we read in from the PWSCF file

    Raises
    ------
    IOError
        Could not find number of electrons in output file.

    """

    nelec, efermi = -1, None
    nbands, nkpts = -1, -1
    alat = -1.0

    # Assume restricted spin treatment until told otherwise
    nspins = 1
    with open(outfile, 'r', encoding='ascii') as f:
        # Find the variables that would not be in the
        # postprocesing file produced by bands.x.
        for ln in f:
            ln_strip = ln.strip()
            if ln_strip.startswith('number of electrons'):
                # HACK: Format descriptor in QE is set to float
                # so read float and then turn into int.
                nelec = int(float(ln.split()[-1]))
            elif ln_strip.startswith('number of Kohn-Sham states'):
                nbands = int(ln.split()[-1])
            elif ln_strip.startswith('number of k points='):
                nkpts = int(ln.split('number of k points=')[-1].split()[0])
            elif ln_strip.startswith('the Fermi energy is'):
                # NB: Fermi energy is ill-defined for insulators
                # It also is not written in the bands post-processing file.
                efermi = float(ln.split()[-1])
            elif re.search('.*SPIN UP|SPIN DOWN', ln_strip):
                nspins = 2
            elif ln_strip.startswith('lattice parameter (alat)'):
                alat = float(ln.split()[-2]) * Bohr

        if nelec == -1:
            raise IOError(
                'Could not find number of electrons in QE output file'
            )
        if nbands == -1:
            raise IOError(
                'Could not find number of bands in QE output file'
            )
        if nkpts == -1:
            raise IOError(
                'Could not find number of k-points in QE output file'
            )
        if alat == -1.0:
            raise IOError(
                'Could not find lattice scaling factor in QE output file'
            )

        # Now put in a dictionary
        keys = ['alat', 'nelec', 'nbands', 'nkpts', 'efermi', 'nspins']
        vals = [alat, nelec, nbands, nkpts, efermi, nspins]
        prop_dict = dict(zip(keys, vals))

    return prop_dict


def read_bands_from_pwscf(outfile: str, nspins: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the output from a PWSCF bands calculation.

    A bit of weirdly formatted file prioritising form over function!

    Parameters
    ----------
    outfile : str
        main output from a bands calculation using PWSCF.
    nspins : int
        number of spins
          1: spin-degenerate
          2: spin-polarised (collinear)
          4: non-collinear

    Returns
    -------
    kpoints : np.ndarray
        list of kpoints in fractional coordinates
    eigs : np.ndarray
        eigenvalues for a given spin

    Raises
    ------
    AssertionError
        Extra bands present beyond what was indicated in header.
    IOError
        Could not find the number of bands/kpoints in the header.

    """
    nb, nk = -1, -1
    with open(outfile, 'r', encoding='ascii') as f:
        for ln in f:
            ln_strip = ln.strip()
            if ln_strip.startswith('number of Kohn-Sham states'):
                nb = int(ln.split()[-1])
            elif ln_strip.startswith('number of k points='):
                nk = int(ln.split('number of k points=')[-1].split()[0])

            if nb != -1 and nk != -1:
                # Have necessary header information stop here first...
                break

        if nk == -1:
            raise IOError('Could not find number of kpoints in PWSCF output')
        if nb == -1:
            raise IOError('Could not find number of bands in PWSCF output')

        # ...allocate data arrays before continuing.
        kpoints = np.empty((nk, 3))
        eigs = np.empty((nspins, nb, nk))

        def _read_eigs_spin(startstr: str, spin_indx: int, file):
            start_bands, have_kpt = False, False
            ib, ik = 0, 0
            for ln in file:
                # Do not start reading until we reach start of band output
                ln = ln.strip()
                if re.search(startstr, ln):
                    start_bands = True
                    continue
                if start_bands is False:
                    continue

                if ik == nk:
                    # Have all kpoints so finish now
                    break

                if re.search('^k =.*bands', ln.strip()):
                    # On k-point line so read coordinates
                    kpoints[ik, :] = ln.split('k =')[1].split()[:3]
                    have_kpt = True
                    ib = 0  # re-initialise band counter

                # Read eigenvalues for the current kpoint
                if have_kpt is True:
                    # Skip the blank line
                    if ln.strip() == '':
                        continue
                    # Still on kpoint line so continue
                    if 'k' in ln:
                        continue
                    # Read eigenvalues on current the line
                    eig_on_ln = np.array(ln.split(), dtype=float)
                    # print(f'{ib=} {ln.strip()=} {len(eig_on_ln)=}')
                    eigs[spin_indx, ib:ib+len(eig_on_ln), ik] = eig_on_ln
                    ib += len(eig_on_ln)

                    # Finished reading all eigenvalues so
                    # reset the flag until we get to the next kpoint.
                    if ib == nb:
                        have_kpt = False
                        ik += 1
                        continue
                    if ib > nb:
                        raise AssertionError(
                            'You have a serious problem, more bands than expected ib>nb'
                        )

            if start_bands is False:
                raise IOError('Could not find bands in file')
            if ik != nk:
                raise AssertionError(
                    'Number of kpoints read does not match header'
                )

        if nspins == 2:
            _read_eigs_spin('.*SPIN UP', 0, f)
            _read_eigs_spin('.*SPIN DOWN', 1, f)
        else:
            _read_eigs_spin('End of band structure calculation', 0, f)
    return kpoints, eigs


def read_bands_from_pp(bandsfile: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the datafile produced by QE bands.x postprocessing script.

    A bit of weirdly formatted file prioritising form over function!
    This file contains the eigenvalues and kpoints for a SINGLE kpoint.
    Thus this function would have to be called twice on two different
    files to get both spin eigenvalues.
    There is no way of telling what the spin component is from the
    file alone...

    Parameters
    ----------
    bandsfile : str
        postprocessing file

    Returns
    -------
    kpoints : np.ndarray
        list of kpoints in fractional coordinates
    eigs : np.ndarray
        eigenvalues for a given spin

    Raises
    ------
    AssertionError
        Extra bands present beyond what was indicated in header.
    IOError
        Could not find the number of bands/kpoints in the header.

    """

    nb, nk = -1, -1
    with open(bandsfile, 'r', encoding='ascii') as f:
        for ln in f:
            # Read the number of kpoints and bands at each
            # kpoint from the header. Continue reading until we reach it
            if ln.strip().startswith('&plot nbnd'):
                header = ln.split()
                nb, nk = int(header[2].strip(',')), int(header[4])
                break

        if nb == -1 or nk == -1:
            raise IOError(
                'Could not find number of bands/kpoints in header'
            )

        # Now read in kpoints and eigenvalues
        ib, ik = -1, -1
        kpoints = np.empty((nk, 3))
        eigs = np.empty((nb, nk))
        for ln in f:
            split_ln = ln.split()
            if ik == nk:  # Since we started on zero indexing
                # Read all kpoints
                break
            if ib == -1:
                # We are on a kpoint line so read it.
                ik += 1
                ib = 0  # Reset band counter to start reading bands
                kpoints[ik] = np.array(split_ln, dtype=float)
            else:
                # Reading in the bands. This is a bit of a pain due to
                # the weird file format. It is not just one eigenvalue per
                # kpoint (that would be too simple). We have to count
                # how many eigenvalues are on the current line and then read them in.
                nb_on_line = len(split_ln)
                eigs[ib:ib + nb_on_line, ik] = np.array(split_ln, dtype=float)

                # Increase band counter before continuing and then check
                # if we have finished reading all bands for this kpoint.
                ib += nb_on_line
                if ib == nb:
                    # Finished reading all bands for kpoint so reset counter
                    # to start reading next kpoint coordinate.
                    ib = -1
                elif ib > nb:
                    raise AssertionError(
                        'You have a serious problem, extra bands present ib>nb'
                    )

        # DEBUG KPOINTS AND EIGENVALUES
        # for ik in range(0, nk):
        #     print(
        #         f'k-point {ik+1} {kpoints[ik, 0]:10.6f}{kpoints[ik, 1]:10.6f}{kpoints[ik, 2]:10.6f}'
        #     )
        # print(
        #     f'k-point 5 {kpoints[4, 0]:10.6f}{kpoints[4, 1]:10.6f}{kpoints[4, 2]:10.6f}')
        # print('Eigenvalues: ', eigs[:, 4])

        return kpoints, eigs


def parse_xml_logical(subtree: ET.Element | None, tag: str, label: str) -> bool:
    """Read the value of a Fortran logical variable in the XML file under a given subtree.

    Parameters
    ----------
    subtree : ET.Element
        subtree containing tag within main QE XML tree.
    tag : str
        XML tag
    label : str
        label for error messages (need not be identical to tag)

    Returns
    -------
    flag : bool
        value of logical variable

    Raises
    ------
    IndexError
        Could not find the tag in the provided subtree.
    ValueError
        Improperly formatted tag value. Must be true or false.
    """

    flag = False
    if subtree.find(tag) is not None:
        tag_val = subtree.find(tag).text.strip().lower()
        if tag_val == 'true':
            flag = True
        elif tag_val == 'false':
            flag = False
        else:
            raise ValueError(f'Unknown value "{tag_val}" ' +
                             f'for {label} - must be "true" or "false".')
    else:
        raise IndexError(f'Could not find "{label}" flag in XML file')
    return flag


def parse_xml_number(subtree: ET.Element | None, tag: str, label: str) -> float:
    """Read the value of a Fortran real/integer variable in the XML file.

    Parameters
    ----------
    subtree : ET.Element
        subtree containing tag within main QE XML tree.
    tag : str
        XML tag
    label : str
        label for error messages (need not be identical to tag)

    Returns
    -------
    val : float
        value of variable

    Raises
    ------
    IndexError
        Could not find the tag in the provided subtree.
    """
    val = float('inf')
    if subtree.find(tag) is not None:
        val = float(subtree.find(tag).text)
    else:
        raise IndexError(f'Could not find "{label}" flag in XML file')
    return val
