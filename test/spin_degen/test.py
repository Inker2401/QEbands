import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spglib
from ase.units import _amu

from QEbands.bands import BandData

AMU_GRAMS = _amu * 1000  # atomic mass units to grams
ANGSTROM_CM = 1e-8  # angstrom to cm


def summarise_properties(mybands: BandData):
    spg_cell = (mybands.cell.cell[:], mybands.cell.get_scaled_positions(),
                mybands.cell.get_atomic_numbers())
    tot_mass = np.sum(mybands.cell.get_masses())
    vol = mybands.cell.get_volume()

    celldens = tot_mass/vol * AMU_GRAMS / ANGSTROM_CM**3  # gm/cm^3

    print(' '*12+'Real lattice (A)')
    for vec in mybands.cell.cell[:]:
        print(f'{vec[0]:12.6f}{vec[1]:12.6f}{vec[2]:12.6f}')
    print('Space group of crystal: ', spglib.get_spacegroup(spg_cell))
    print(f'Cell density: {celldens:.7f} g/cm^3')
    print(' ')

    print(f'Number of electrons: {mybands.nelec:6n}')
    print(f'Number of bands:     {mybands.nbands:6n}')
    print(f'Number of spins:     {mybands.nspins:6n}')
    print(f'Spin treatment:      {mybands.spin_treat}')
    print(f'Fermi energy: {mybands.efermi:12.6f} eV')

    print('Path labels', mybands.high_sym_labels)

    print('Eigenvalues (eV)')
    print('k-point 1    ', mybands.eigenvalues[0, :, 0])
    print('k-point 2    ', mybands.eigenvalues[0, :, 1])
    print('k-point nk-1 ', mybands.eigenvalues[0, :, -2])
    print('k-point nk   ', mybands.eigenvalues[0, :, -1])


# Try to read from XML
print('XML read')
bands = BandData('pwscf.xml', zero_fermi=False)
summarise_properties(bands)
print('')

# Try to read from PWSCF output
print('PWSCF read')
bands = BandData('Si.bands.out', zero_fermi=False)
summarise_properties(bands)
print('')

# Try to read from bands data file
print('PP read')
bands = BandData('Si.bands.out', 'Si.bands.dat', zero_fermi=False)
summarise_properties(bands)
print('')
