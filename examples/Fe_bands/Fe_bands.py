import matplotlib as mpl
import matplotlib.pyplot as plt

from QEbands.bands import BandData

# Unless explicitly stated as an option, plot parameters are controlled via rcParams
mpl.rcParams.update({'font.size': 14,
                     'lines.linewidth': 1.75,
                     'lines.markersize': 10
                     })

elim = [-10, 20]

# Initialise the bands data - will initialise such that energy scale has Fermi enery is zero
bands = BandData('Fe.xml', zero_fermi=True, use_spg_labels=True)


def plot_bands(mybands: BandData, spin_indx: int,
               start: int, end: int, color: str,
               label: str = ''):
    """Plot a set of bands for a given spin"""
    for nb in range(start, end):
        if nb == start:
            plt.plot(mybands.eigenvalues[spin_indx, nb, :], c=color, label=label)
        else:
            plt.plot(mybands.eigenvalues[spin_indx, nb, :], c=color)


# Plot the bands by spin polarisation
fig, ax = plt.subplots()
bands.setup_axes(ax)

plot_bands(bands, 0, 0, bands.nbands, color='b', label='spin up')
plot_bands(bands, 1, 0, bands.nbands, color='r', label='spin down')
ax.set_ylim(elim)
ax.legend()

plt.savefig('Fe_bands_by_spin.png')
plt.show()
