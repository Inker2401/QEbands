import matplotlib as mpl
import matplotlib.pyplot as plt

from QEbands.bands import BandData

# Unless explicitly stated as an option, plot parameters are controlled via rcParams
mpl.rcParams.update({'font.size': 14,
                     'lines.linewidth': 1.75,
                     'lines.markersize': 10
                     })

# Initialise the bands data - will initialise such that energy scale has Fermi enery is zero
bands = BandData('Si.xml', zero_fermi=True)

# Create the figure axis
fig, ax = plt.subplots()
bands.setup_axes(ax)

# Plot eigenvalues (occupied)
for ns in range(bands.nspins):
    for nb in range(bands.nelec//bands.occ):
        plt.plot(bands.eigenvalues[ns, nb, :], c='b')

# Plot eigenvalues (unoccupied)
for ns in range(bands.nspins):
    for nb in range(bands.nelec//bands.occ, bands.nbands):
        plt.plot(bands.eigenvalues[ns, nb, :], c='r')

# Set energy scale
ax.set_ylim([-15, 15])

plt.savefig('Si_bands.png')

plt.show()
