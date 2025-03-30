"""
This module handles the plotting of a QE band structure.
Most of the bands data from a PWSCF calculation
is stored in a BandData instance.

Plot parameters are controlled from matplotlib.rcParams
for the most part unless stated otherwise.

Author: V Ravindran 02/04/2025

# TODO Non-collinear support?
# TODO Get distance between k-points?
"""
import xml.etree.ElementTree as ET
from typing import Optional
from warnings import warn

import ase
import ase.io as aseio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase.units import Bohr, Hartree

import QEbands.ioutils as io
import QEbands.symutils as sym


class BandData:
    """
    Data from a band structure calculation in the PWSCF of Quantum ESPRESSO.

    Unless otherwise stated, the units in this class FOLLOWING initialisation are
    angstroms for lengths and electronvolts (eV) for energy.

    Attributes
    ----------
    spin_treat : str
        string stating spin treatment for calculation ('DEGENERATE', 'COLLINEAR', 'NON-COLLINEAR')
    nspins : int
        number of spins (1 - degenerate, 2 - collinear, 4 - non-collinear)
    nelec : int
        total number of electrons
    nelec_spin : np.ndarray
        number of electrons in each spin channel
    nkpts : int
        number of k-points
    nbands : int
        number of Kohn-Sham states / bands
    eigenvalues : np.ndarray
        Kohn-Sham eigenvalues (shape: nspins,nbands,nkpts)
    occ : int
        occupancy of each Kohn-Sham orbital (2 if DEGENERATE spin treatment, 1 otherwise)
    efermi : float
        Fermi energy
    cell : ase.Atoms
        the structure used for the calculation

    Methods
    -------
    shift_bands:
        shifts the energy scale (i.e. both Fermi energy and eigenvalues) by a rigid amount
    setup_axes:
        sets up the high-symmetry axis for a matplotlib plot
    bandgap:
        returns a dictionary containing information about the bandgap.
    """

    def __init__(self,
                 outfile: str,
                 ppfile_up: Optional[str] = None,
                 ppfile_down: Optional[str] = None,
                 zero_fermi: bool = True,
                 use_spg_labels: bool = True,
                 sym_method: str = 'proj',
                 sym_tol: Optional[float] = None,
                 ):
        """Data from a band structure calculation in the PWSCF of Quantum ESPRESSO.

        This class may be initialised in several ways:
        1. the XML file from a bands calculation
        2. the PWSCF output from a bands calculation (requires high verbosity)
        3. the PWSCF output + post-processed file produced by bands.x
           if doing a spin degenerate calculation, either specify ppfile_up OR ppfile_down (but not both!)

        Note nelec_spin necessary for some functionality relies on occupancies which are
        only written to the XML. This can, of course, be overriden in the class directly.

        Parameters
        ----------
        outfile : str
            the PWSCF output from a bands calculation.
        ppfile_up : Optional[str]
            the output file produced by bands.x (for up spins if spin-polarised)
        ppfile_down : Optional[str]
            the output file produced by bands.x (for down spins if spin-polarised)
        zero_fermi : bool
            set energy scale such that Fermi energy is at 0 eV.
        use_spg_labels : bool
            use high-symmetry labels based on space group rather than just Bravais lattice
            (Default: True)
        sym_method : str
            method for determining high-symmetry points.
            See documentation of symutils.get_high_sym_points for more details.
        sym_tol : Optional[float]
            tolerance for determination of high-symmetry points
            See documentation of symutils.get_high_sym_points for more details.
        """

        self.spin_treat = ''
        self.nspins = -1
        self.nelec = -1
        self.nelec_spin = np.array([-1], dtype=int)
        self.nkpts = -1
        self.nbands = -1
        self.eigenvalues = np.array([], dtype=float)
        self.occ = -1
        self.kpoints = np.array([], dtype=float)
        self.efermi = None
        self.cell = None

        # Determine whether we have an XML file or not.
        # If we do it makes things considerably easier...
        have_xml = outfile.endswith('.xml')
        if have_xml:
            _read_xml_props(self, outfile)
        else:
            # Get the data from a either the PWSCF output file or post-processed files.
            _read_pwscf_props(self, outfile, ppfile_up, ppfile_down)

            # Occupancies are not written to the output file. V Ravindran 08/04/2025
            # If spin-degenerate, just set to no of electrons.
            if self.nspins == 1:
                self.nelec_spin = np.array([self.nelec], dtype=int)

        # Set the occupancies
        self.occ = 1
        if self.nspins == 2:
            self.spin_treat = 'COLLINEAR'
        else:
            self.spin_treat = 'RESTRICTED'
            self.occ = 2  # double occupancy

        # Sort out Fermi energy if needed.
        # Fermi energy is not defined for an insulator so get it VBM
        if self.efermi is None:
            if self.nspins == 2 and np.any(self.nelec_spin == -1):
                warn('nelec_spin not initialised, using total electrons instead to find Fermi energy - will likely be total nonsense!')
                vbm = self.eigenvalues[0, self.nelec//self.occ-1, :]
            else:
                vbm = self.eigenvalues[0, self.nelec_spin[0]//self.occ-1, :]
            self.efermi = np.max(vbm)

        # Identify the Bravais lattice
        if use_spg_labels is True:
            # Assign based on spacegroup rather than just Bravais lattice
            bv = sym.get_bravais_lattice_spg(self.cell)
        else:
            bv = self.cell.cell.get_bravais_lattice()

        # Get the high-symmetry points
        self.high_sym_ticks, self.high_sym_labels = sym.get_high_sym_points(
            self.kpoints, bv, method=sym_method, sym_tol=sym_tol
        )

        # Set the energy scale
        if zero_fermi is True:
            self.shift_energy(-self.efermi)

    def shift_energy(self, shift: float):
        """Shift the energy scale by a rigid amount"""
        self.efermi += shift
        self.eigenvalues += shift

    def setup_axes(self, ax: mpl.axes._axes.Axes,
                   sym_labelsize: float = 20,
                   eng_labelsize: float = 16,
                   fermi_lw: float = 1.75,
                   fermi_ls: str = '--',
                   fermi_c: str = '0.5',
                   frame_edgewidth: float = 1.5,
                   high_sym_lw: float = 1.75
                   ):
        """Sets up a band structure plot axes.

        This adds the high-symmetry lines and the Fermi energy.

        Parameters
        ----------
        ax : mpl.axes._axes.Axes
            axis to use for plot
        sym_labelsize : float
            label fontsize for high-symmetry points
        eng_labelsize : float
            label fontsize for energy ticks
        fermi_lw : float
            linewidth for Fermi energy
        fermi_ls : str
            linestyle for Fermi energy
        fermi_c : str
            colour for Fermi energy
        frame_edgewidth : float
            edgewidth for plotting frame
        high_sym_lw : float
            linewidth for high-symmetry lines
        """

        # Add the high-symmetry lines to the plots
        kticks, klabels = self.high_sym_ticks, self.high_sym_labels
        ax.set_xticks(kticks)
        ax.set_xticklabels(klabels, fontsize=sym_labelsize)

        # Add the Fermi energy
        ax.axhline(self.efermi, c=fermi_c, ls=fermi_ls, lw=fermi_lw)

        # Adjust energy axis to have the same font size.
        ax.set_ylabel(r'$E$ (eV)', fontsize=eng_labelsize)
        ax.set_xlim(0, len(self.kpoints)-1)

        # Adjust tick sizes
        ax.tick_params(axis='both', direction='in', which='major',
                       labelsize=eng_labelsize,
                       length=12, width=1.2)
        ax.tick_params(axis='both', which='minor', direction='in', length=6,
                       right=True, left=True, top=False, bottom=False)
        ax.minorticks_on()

        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

        # P. Dirk 08/04/2024: Add high symmetry lines but only if not duplicated.
        for i in range(1, len(kticks)):
            if kticks[i] - kticks[i-1] > 1:
                ax.axvline(kticks[i], c='k', ls='-', lw=high_sym_lw)

        # Adjust frame thickness
        for x in ax.spines.values():
            x.set_linewidth(frame_edgewidth)

    def bandgap(self) -> dict:
        """Returns a dictionary containing information about the bandgap."""
        if self.nspins != 1 and (self.nelec_spin == -1).any():
            raise AssertionError('Do not have electrons for each spin channel - reinitialise from XML or provide manually')
        # Get the valence and conduction states
        vb_eigs = np.empty((self.nspins, self.nkpts))
        cb_eigs = np.empty((self.nspins, self.nkpts))
        for ns in range(self.nspins):
            vb_eigs[ns] = self.eigenvalues[ns, int(self.nelec_spin[ns]/self.occ) - 1, :]
            cb_eigs[ns] = self.eigenvalues[ns, int(self.nelec_spin[ns]/self.occ), :]

        # Determine valence band maximum and conduction band minimum.
        vbm_i = np.argmax(vb_eigs, axis=1)
        cbm_i = np.argmin(cb_eigs, axis=1)
        vbm_eig = np.max(vb_eigs, axis=1)
        cbm_eig = np.min(cb_eigs, axis=1)

        loc_vbm = np.empty((self.nspins, 3))
        loc_cbm = np.empty((self.nspins, 3))

        # Calculate direct and indirect gap
        gap_in = cbm_eig - vbm_eig
        gap_dir = np.empty(self.nspins)
        for ns in range(self.nspins):
            gap_dir[ns] = cb_eigs[ns, vbm_i[ns]] - vb_eigs[ns, vbm_i[ns]]
            # Sneak in kpoint of VBM and CBM here
            loc_vbm[ns] = self.kpoints[vbm_i[ns]]
            loc_cbm[ns] = self.kpoints[cbm_i[ns]]

        # Put it all in a dictionary
        gap_info = {
            'vbm_indx': vbm_i, 'cbm_indx': cbm_i,
            'vbm_eig': vbm_eig, 'cbm_eig': cbm_eig,
            'gap_in': gap_in, 'gap_dir': gap_dir,
            'loc_vbm': loc_vbm, 'loc_cbm': loc_cbm
        }

        return gap_info


def _read_xml_props(mybands: BandData, xmlfile: str):
    """Initialise the the BandsData from an QE XML file."""
    with open(xmlfile, 'r', encoding='ascii') as f:
        tree = ET.parse(f)

    # Let's make sure we have a band structure
    calc_type = tree.find('.//calculation').text.strip().lower()
    if calc_type != 'bands':
        raise AssertionError('PWSCF did not run a "bands" calculation')

    # Now let's start extracting the appropriate bits we need
    bs_elem = tree.find('.//band_structure')

    spin_pol = io.parse_xml_logical(bs_elem, 'lsda',
                                    'spin polarisation')
    have_ncm = io.parse_xml_logical(bs_elem, 'noncolin',
                                    'non-collinear')
    mybands.nspins = 1
    if spin_pol is True:
        mybands.nspins = 2

    if have_ncm is True:
        raise NotImplementedError(
            'Non-collinear band structures read for XML unsupported'
        )

    mybands.efermi = io.parse_xml_number(bs_elem, 'fermi_energy',
                                         'fermi energy')

    mybands.nelec = int(io.parse_xml_number(bs_elem, 'nelec',
                                            'no. of of electrons'))
    mybands.nkpts = int(io.parse_xml_number(bs_elem, 'nks',
                                            'no of kpoints'))
    if mybands.nspins == 2:
        nb_up = int(io.parse_xml_number(bs_elem, 'nbnd_up',
                                        'no of spin up bands'))
        nb_down = int(io.parse_xml_number(bs_elem, 'nbnd_dw',
                                          'no of spin down bands'))
        mybands.nbands = max(nb_up, nb_down)
    else:
        mybands.nbands = int(io.parse_xml_number(bs_elem, 'nbnd',
                                                 'no of Kohn-Sham states'))

    # Now we read the k-points and their band energies.
    # NB: These are in units of 2*Pi/alat so they will need to be converted
    # (done below once we have the cell).
    # While we're in here, we can also grab the occupancies. V Ravindran 08/04/2024
    mybands.kpoints = np.empty((mybands.nkpts, 3), dtype=float)
    mybands.eigenvalues = np.empty((mybands.nspins, mybands.nbands, mybands.nkpts),
                                   dtype=float)
    occ_array = np.empty((mybands.nspins, mybands.nbands, mybands.nkpts), dtype=float)
    for ik, kpt in enumerate(bs_elem.findall('ks_energies')):
        coord = kpt.find('k_point').text.split()
        eigs = np.array(kpt.find('eigenvalues').text.split(),
                        dtype=float)
        occs_at_kpt = np.array(kpt.find('occupations').text.split(),
                               dtype=float)
        mybands.kpoints[ik] = coord
        # Loop around spins - the eigenvalues are together for both spins
        # so make sure to advance accordingly.
        for ns in range(mybands.nspins):
            eig_start = ns * mybands.nbands
            eig_end = (ns+1) * mybands.nbands
            mybands.eigenvalues[ns, :, ik] = eigs[eig_start: eig_end]
            occ_array[ns, :, ik] = occs_at_kpt[eig_start: eig_end]

    # Analyse the occupancy array to determine number of electrons per spin channel. V Ravindran 08/04/2025
    # Naturally, we only need to do this if spin-polarised. V Ravindran 08/04/2025
    if mybands.nspins == 2:
        mybands.nelec_spin = analyse_occ_array(mybands.nspins, mybands.nkpts,
                                               mybands.nbands, occ_array)
        mybands.nelec_spin += 1
    else:
        mybands.nelec_spin = np.array([mybands.nelec], dtype=int)

    # NB The XML data is in Hartrees so make sure to convert
    # the Fermi energy and the eigenvalues to eV.
    if mybands.efermi is not None:
        mybands.efermi *= Hartree
    mybands.eigenvalues *= Hartree

    # Now read and construct the cell
    cell_info = tree.find('.//atomic_structure')
    lat_info = cell_info.find('cell')
    lat_vecs = np.empty((3, 3), dtype=float)
    for i in range(3):
        vec = lat_info.find(f'a{i+1}').text.split()
        lat_vecs[i] = vec

    natoms = int(cell_info.items()[0][-1])
    atom_species = np.empty(natoms, dtype='<U2')
    atom_pos = np.empty((natoms, 3))
    atom_info = cell_info.find('atomic_positions')
    for i, elem in enumerate(atom_info.iter()):
        if i == 0:
            # Skip header
            continue
        atom_species[i-1] = elem.items()[0][-1]
        atom_pos[i-1] = elem.text.split()

    # Get scaling factor used by PWSCF (alat)
    alat = float(cell_info.items()[1][-1])

    # NB: The XML data is in Bohr for the lattice vectors and positions so make sure to convert
    lat_vecs *= Bohr
    atom_pos *= Bohr
    alat *= Bohr

    mybands.cell = ase.Atoms(
        symbols=atom_species,
        positions=atom_pos,
        cell=lat_vecs, pbc=True
    )

    # Now convert k-points to fractional coordinates.
    mybands.kpoints = sym.convert_kpts_tpiba_to_crystal(mybands.cell.cell[:],
                                                        alat,
                                                        mybands.kpoints)


def _read_pwscf_props(mybands: BandData,
                      outfile: str,
                      ppfile_up: Optional[str] = None,
                      ppfile_down: Optional[str] = None
                      ):
    """Initialise the the BandsData from an QE XML file."""
    # At a bare minimum, we will require the output file from PWSCF
    # as this is not contained in the ppfiles.
    prop_dict = io.read_pwscf_header(outfile)
    mybands.nelec = prop_dict['nelec']
    mybands.nspins = prop_dict['nspins']
    mybands.efermi = prop_dict['efermi']
    mybands.nkpts = prop_dict['nkpts']
    mybands.nbands = prop_dict['nbands']
    alat = prop_dict['alat']

    # Pre-allocate arrays - this will facilitate cross checks
    # (IndexError will raise if passing in shape mismatch)
    mybands.eigenvalues = np.empty(
        (mybands.nspins, mybands.nbands, mybands.nkpts)
    )
    mybands.kpoints = np.empty((mybands.nkpts, 3))

    # Now read eigenvalues, we either have to get it from the output
    # (only possible on high verbosity) or the post-processed file.
    if mybands.nspins == 2:
        # Collinear spin has two post-processed files, if neither are provided,
        # read from PWSCF output file.
        if ppfile_up is None and ppfile_down is None:
            kpts, eigs = io.read_bands_from_pwscf(
                outfile,
                mybands.nspins
            )
            mybands.kpoints[:, :], mybands.eigenvalues[:, :, :] = kpts, eigs
        else:
            # Read from post-processed files instead
            if ppfile_up is None:
                raise FileNotFoundError('Missing ppfile_up')
            if ppfile_down is None:
                raise FileNotFoundError('Missing ppfile_down')
            kpts, eigs = io.read_bands_from_pp(ppfile_up)
            mybands.kpoints = kpts
            mybands.eigenvalues[0, :, :] = eigs
            kpts, eigs = io.read_bands_from_pp(ppfile_down)
            mybands.eigenvalues[1, :, :] = eigs
    else:
        # We either require a single post-processed file or the PWSCF output.
        if ppfile_up is not None and ppfile_down is not None:
            raise ValueError(
                'Specify either ppfile_up and ppfile_down for spin-degenerate.'
            )
        if ppfile_up is not None or ppfile_down is not None:
            if ppfile_up is not None:
                kpts, eigs = io.read_bands_from_pp(ppfile_up)
            elif ppfile_down is not None:
                kpts, eigs = io.read_bands_from_pp(ppfile_down)
            mybands.kpoints[:, :] = kpts
            mybands.eigenvalues[0, :, :] = eigs
        else:
            # Neither post-processed file provided, read from PWSCF
            kpts, eigs = io.read_bands_from_pwscf(outfile, mybands.nspins)
            mybands.kpoints = kpts
            mybands.eigenvalues = eigs

    # Read the cell
    mybands.cell = aseio.read(outfile)

    # k-points in PWSCF for band output are written in units of 2pi/alat so we will
    # need to convert them.
    mybands.kpoints = sym.convert_kpts_tpiba_to_crystal(mybands.cell.cell[:],
                                                        alat,
                                                        mybands.kpoints)


def analyse_occ_array(nspins: int, nkpts: int, nbands: int, occ_array: np.ndarray,
                      occ_tol: float = 1e-12) -> np.ndarray:
    # TODO This sometimes doesn't count correctly.
    """Analyse the occupancy array and find the highest occupied bands.

    This simply counts occupancies from lowest to highest bands stopping once we reach an unoccupied
    band in a given spin channel. Note this will probably not work with some smearing schemes due to
    partial occupancies.

    Parameters
    ----------
    nspins : int
        no. of spins
    nkpts : int
        no. of kpoints
    nbands : int
        no. of eigenvalues (total if spin-degenerate, for each spin channel if spin-polarised)
    occ_array : np.ndarray
        occupancy of each Kohn-Sham orbital
    occ_tol : float
        tolerance for determining if a band is occupied

    Returns
    -------
    np.ndarray
        highest (fully) occupied band index for each spin channel

    Raises
    ------
    IndexError
        occ_array is wrong shape

    """

    if occ_array.shape != (nspins, nbands, nkpts):
        raise IndexError('Occupancy array does not have the correct shape, expected: ' +
                         f'({nspins},{nkpts}{nbands}) but obtained {occ_array.shape}')

    highest_occ = np.empty(nspins, dtype=int)

    # Find highest occupied band for each spin channel.
    # V Ravindran: In principle, we could speed this up by looking
    # around near the Fermi level.
    for ns in range(nspins):
        for nb in range(nbands):
            if (np.abs(occ_array[ns, nb, :]-1.0).any() > occ_tol):
                break
            else:
                highest_occ[ns] = nb

    return highest_occ
