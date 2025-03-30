"""
Handles symmetry related operations such as determination
of high-symmetry points.
Author: V Ravindran 01/04/2025
"""
from typing import Optional

import ase
import ase.lattice as latt
import numpy as np
import spglib


def get_bravais_lattice_usr(cell: ase.Atoms, bv_type: str) -> latt.BravaisLattice:
    """Initialise the Bravais lattice for lattice parameters for the user's cell.

    Parameters
    ----------
    cell : ase.Atoms
        atomic structure used in calculation.
    bv_type : str
        3 letter Bravais lattice label

    Returns
    -------
    bv : ase.lattice.BravaisLattice
        Bravais lattice for structure.

    Raises
    ------
    IndexError
        Unknown Bravais specified.
    """
    # Get all the Bravais lattices
    bv_dict = latt.bravais_lattices

    # Get the cell's lattice parameters
    a, b, c, alpha, beta, gamma = cell.cell.cellpar()

    # Here begins the boring bit - there should be 14 Bravais lattices here!
    # Since the ASE interface is not overloaded, we will have to
    # set the arguments ourselves here.
    # Lattice Parameters not specified have values implied by type of Bravais lattice.
    if bv_type == 'TRI':  # Triclinic lattice
        bv = bv_dict[bv_type](a=a, b=b, c=c,
                              alpha=alpha, beta=beta, gamma=gamma)
    elif bv_type in ('MCL', 'MCLC'):  # Primitive/Base-centred (C-centred) Monoclinic
        bv = bv_dict[bv_type](a=a, b=b, c=c,
                              alpha=alpha)
    # Primitive or Body-/Face-Centred or A/C-Centred Orthorhombic
    elif bv_type in ('ORC', 'ORCI', 'ORCF', 'ORCC'):
        bv = bv_dict[bv_type](a=a, b=b, c=c)
    elif bv_type in ('TET', 'BCT'):  # Primitive/Body-Centred Tetragonal
        bv = bv_dict[bv_type](a=a, c=c)
    elif bv_type == 'RHL':  # R-trigonal/Rhombohedral
        bv = bv_dict[bv_type](a=a, alpha=alpha)
    elif bv_type == 'HEX':  # Hexagonal
        bv = bv_dict[bv_type](a=a, c=c)
    # Primitive/Simple or Body-Centred or Face-Centred Cubic
    elif bv_type in ('CUB', 'BCC', 'FCC'):
        bv = bv_dict[bv_type](a=a)
    else:
        # Unless someone's reinvented crystallography and how 3D space works...
        raise IndexError(f'Unknown Bravais lattice: {bv_type}')

    return bv


def get_bravais_lattice_spg(cell: ase.Atoms) -> ase.lattice.BravaisLattice:
    """Determine the high-symmetry points using the spacegroup.

    For low-symmetry Bravais lattices where the special/high-symmetry points
    are lattice parameter dependent, we will use the lattice parameters
    of the computational cell.
    In magnetic structures, the breaking of crystal symmetry by e.g.
    antiferromagnetic ordering can result in a primitive cell that is
    not correctly recognised by its crystallographic symmetry by ASE.

    Parameters
    ----------
    cell : ase.Atoms
        atomic structure used in calculation.

    Returns
    -------
    bv : ase.lattice.BravaisLattice
        Bravais lattice for structure.

    Raises
    ------
    IndexError
        Ended up finding Bravais lattice that does not match space group.
    """
    # spglib expects cell as a tuple with in the order of
    # lattice vectors, fractional coords and species (by atomic number)
    spg_cell = (cell.cell[:], cell.get_scaled_positions(),
                cell.get_atomic_numbers())

    # Get the space group information for this cell
    spg_symb, spgno_str = spglib.get_spacegroup(spg_cell).split()
    # Remove the brackets returned around number in the above
    spg_no = int(spgno_str[spgno_str.find('(') + 1: spgno_str.find(')')])

    # Get the first letter of the spacegroup in international notation.
    # This gives the information about the Bravais lattice
    bv_symb = spg_symb[0]

    # Now use the space group to determine the crystal system.
    # We can determine the actual Bravais lattice using the first
    # letter of the international notation symbol.
    #
    # Particularly for low symmetry Bravais lattices
    # where the high symmetry points depend on lattice parameters,
    # we will use the computational cell's lattice parameters.
    # This deviates slightly from the convention used by
    # M. Setyawan., S. Curtarolo Mat. Sci. 49 (2010) 299–312 (2010)
    # where the high-symmetry points are defined for the pirmitive cell by convention.
    if 1 <= spg_no <= 2:
        # Triclinic lattice
        bv_type = 'TRI'
    elif 3 <= spg_no <= 15:
        # Monoclinic
        if bv_symb == 'P':  # Primitive monoclinic
            bv_type = 'MCL'
        elif bv_symb == 'C':  # Base-centred (C-centred) monoclinic
            bv_type = 'MCLC'
        else:
            raise IndexError(
                f'Unknown monoclinic lattice with space group: {spg_symb}')
    elif 16 <= spg_no <= 74:
        # Orthorhombic
        if bv_symb == 'P':  # Primitive Orthorhombic
            bv_type = 'ORC'
        elif bv_symb == 'I':  # Body-Centred Orthorhombic
            bv_type = 'ORCI'
        elif bv_symb == 'F':  # Face-Centred Orthorhombic
            bv_type = 'ORCF'
        elif bv_symb in ('A', 'C'):  # A/C-centred Orthorhombic
            bv_type = 'ORCC'
        else:
            raise IndexError(
                f'Unknown orthorhombic lattice with space group: {spg_symb}')
    elif 75 <= spg_no <= 142:
        # Tetragonal
        if bv_symb == 'P':  # Primitive Tetragonal
            bv_type = 'TET'
        elif bv_symb == 'I':  # Body-Centred Tetragonal
            bv_type = 'BCT'
        else:
            raise IndexError(
                f'Unknown tetragonal lattice with space group: {spg_symb}')
    elif 143 <= spg_no <= 167:
        # Trigonal
        if bv_symb == 'R':  # R-trigonal/Rhombohedral
            bv_type = 'RHL'
        elif bv_symb == 'P':  # Hexagonal
            bv_type = 'HEX'
        else:
            raise IndexError(
                f'Unknown trigonal lattice with space group: {spg_symb}')
    elif 168 <= spg_no <= 194:
        # Hexagonal
        bv_type = 'HEX'
    elif 195 <= spg_no <= 230:
        # Cubic
        if bv_symb == 'P':  # Primitive/Simple Cubic
            bv_type = 'CUB'
        elif bv_symb == 'I':  # Body-Centred Cubic
            bv_type = 'BCC'
        elif bv_symb == 'F':  # Face-Centred Cubic
            bv_type = 'FCC'
        else:
            raise IndexError(
                f'Unknown cubic lattice with space group: {spg_symb}')
    else:
        raise IndexError(f'Unknown Spacegroup {spg_no}: {spg_symb}')

    # Now get the Bravais lattice
    bv = get_bravais_lattice_usr(cell, bv_type)
    return bv


def get_high_sym_points(kpts: np.ndarray,
                        bv: latt.BravaisLattice,
                        method: str = 'proj',
                        sym_tol: Optional[float] = None) -> tuple[np.ndarray, list]:
    """Finds the high-symmetry points for a given list of kpoints.

    This is based on the idea that given a kpoint, k_i,
    the previous kpoint k_(i-1) and next one
    k_(i+1), if all three kpoints are nearly collinear,
    then they are on the same path.
    Otherwise, this indicates a change of direction and one can
    "reasonably" assume that the current kpoint is a high-symmetry kpoint.

    Parameters
    ----------
    kpts : np.ndarray
        kpoints in the band structure
    bv : latt.BravaisLattice
        Bravais lattice for crystal
    method : str
        heuristic to use for high-symmetry point determination
    sym_tol : float
        tolerance for determination of high symmetry points

    Returns
    -------
    ticks : np.ndarray
        kpoint indices that mark high-symmetry points
    labels : list
        the labels for each tick

    Raises
    ------
    ValueError
    IndexError
    AssertionError


    """

    SAME_PT_TOL = 1e-8

    def _get_high_sym_proj(nkpts: int, kpts: np.ndarray, sym_tol: float):
        # Assume we start and finish on high-symmetry point
        is_high_sym = np.empty(nkpts, dtype=bool)
        is_high_sym[0] = True
        is_high_sym[-1] = True

        # Calculate the position vectors between each successive kpoint.
        dks = np.diff(kpts, axis=0)
        dks_sq = np.sum(dks**2, axis=1)

        # Now check if kpoints lie on top of each other
        # (compare their squares to avoid precision loss).
        if np.any(dks_sq < SAME_PT_TOL):
            raise AssertionError(
                'Have two subsequent kpoints lying on top of each other')

        # Calculate the projection between each position vector
        #     <dk1|dk2>/(|dk1||dk2|) = 1
        # if the points are collinear.
        dots = np.sum(dks[0:nkpts-2] * dks[1:nkpts-1], axis=1)
        denoms = np.sqrt(dks_sq[0:nkpts-2] * dks_sq[1:nkpts-1])
        projs = dots / denoms

        # High symmetry point if large deviation from 1 (not collinear)
        is_high_sym[1:nkpts-1] = np.abs(projs - 1) > sym_tol
        return is_high_sym

    def _get_high_sym_grad(nkpts: int, kpts: np.ndarray, sym_pt_tol: float):
        # Assume we start and finish on high-symmetry point
        is_high_sym = np.empty(nkpts, dtype=bool)
        is_high_sym[0] = True
        is_high_sym[-1] = True

        # Calculate the gradient in kpoints
        gradk = np.diff(kpts, axis=0)
        grad2k = np.diff(gradk, axis=0)

        for i, diff in enumerate(grad2k):
            # The second derivative or change in the gradient has a large
            # change along any kpoint due to a rapid change of direction,
            # current point is likely a high symmetry point.
            #
            # +1 on LHS since we want index of gradk which is 1 behind
            is_high_sym[i+1] = np.any(np.abs(diff) > sym_pt_tol)
        return is_high_sym

    if kpts.shape[-1] != 3 and kpts.ndim != 2:
        raise IndexError('Expected shape of kpts is (nkpts, 3).')
    nkpts = kpts.shape[0]

    if method == 'proj':
        if sym_tol is None:
            sym_tol = 5e-5
        high_sym_mask = _get_high_sym_proj(nkpts, kpts, sym_tol)
    elif method == 'grad':
        if sym_tol is None:
            sym_tol = 1e-5
        high_sym_mask = _get_high_sym_grad(nkpts, kpts, sym_tol)
    else:
        raise ValueError('Unknown high-symmetry method: {method}')

    special_points = bv.get_special_points()

    high_sym = kpts[high_sym_mask, :]
    ticks = np.where(high_sym_mask)[0]
    labels = [' ']*high_sym.shape[0]

    for ik, k in enumerate(high_sym):
        for label in special_points:
            special_pt_coord = special_points[label]
            # P. Dirk 08/04/2025 - absolute value is actually necessary
            # here or it will map onto 1-diff rather than the diff.
            if (np.mod(np.absolute(special_pt_coord - k), 1) < sym_tol).all():
                if label == 'G':
                    labels[ik] = r'$\Gamma$'
                elif label.isalnum() and not label.isalpha():
                    # P. Dirk 08/04/2025 - added subscripts for labels like X1.
                    labels[ik] = rf'${label[:-1]}_{label[-1]}$'
                else:
                    labels[ik] = label
    return ticks, labels


def convert_kpts_tpiba_to_crystal(real_lat: np.ndarray, alat: float, kpts: np.ndarray) -> np.ndarray:
    """Convert k-points from units of 2*pi/a to fractional coordinates.

    First, some notes on conventions since people apparently cannot agree on basic things...
    Mathematically speaking, we simply have the relation between direct lattice and reciprocal lattice vectors:
               <a_i|b_j> = 2*pi delta_ij
    Both sets of lattice vectors can be arranged as a matrix (and this is where people can't agree).
    In most electronic structure codes (and ASE), the matrix of direct lattice vectors, A, is arranged row-wise,
    i.e. a lattice vector on each row.

    For the matrix of reciprocal lattice vectors (also ROW-wise), then we get
               B * A.T = 2*pi * I
    The transpose is necessary because we want a_i to be along the columns when multiplying out.
    ASE uses the convention where the lattice vectors are in units of 2*pi with B returned such that
               B * A.T = I
    such that A.T = B^-1 (as does Quantum ESPRESSO)

    Parameters
    ----------
    real_lat : np.ndarray
        real lattice vectors (vectors on rows)
    alat : float
        scaling factor used by PWSCF.
    kpts : np.ndarray
        list of kpoints (row-ordered)

    Returns
    -------
    np.ndarray
        converted list of kpoints

    """
    # Quantum ESPRESSO will typically output the kpoints in units of 2*pi/alat and direct lattice in terms of alat.
    # Therefore adjust real_lat so we can do the same.
    real_lat = real_lat / alat

    # Note that to get from fractional coordinates p to Cartesian coordinates k, we have
    # k_x = p_x * bj_x  (j=1,3) and so on
    # i.e. as a matrix we require the tranpose of such that we have
    # ki = B_ji * pi
    # i.e. with the components of the recip lattice along the row or equivalently the vectors along the column, i.e. B^T.
    # Hence, (B^T)^-1 = (B^-1).T = A
    for ik, k in enumerate(kpts):
        kpts[ik] = real_lat @ k
    return kpts
