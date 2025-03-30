# QE-bands
A visualisation tool for Quantum ESPRESSO band structures in Python.

This package is inspired heavily by the [CASTEP-bands](https://github.com/NPBentley/CASTEP_bands) package for the CASTEP plane-wave pseudopotential electronic structure code.

## Installation
Installation is done via pip using the following command

```
pip install https://github.com/Inker2401/QEbands.git
```

In addition to the usual scientific Python dependencies (NumPy and Matplotlib), this package also requires the following:
- [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/)
- [Spglib](https://github.com/spglib/spglib/releases)

## Usage
You are encouraged to consult the `examples` folder to see some example plots in some systems.
In addition, bear in mind the following:

- All the data is stored in the `BandData` class. This class may be initialised in the following way (see `test` folder for each possible initialisation)
  1. the Quantum ESPRESSO XML file for a `bands` run (**recommended**)
  2. the standard output of a PWSCF run
  3. the standard output of a PWSCF run with the post-processed output of the `bands.x` utility.

- Note some class attributes are not initialised without an XML file, notably the `nelec_spin` which contains the number of electrons for each spin channel.

- The definitions of high-symmetry points are those of [W. Setyawan and S. Curtarolo, Comp. Mat. Sci. 49 (2010)](http://dx.doi.org/10.1016/j.commatsci.2010.05.010).

- **Very important!* QEbands assumes that the k-points used in a bandstructure are equally spaced apart. This may not be the case if you use the default
  path generation capabilties of Quantum ESPRESSO. Instead it is recommended that you use a tool like the

  The ASE package can generate band structure paths paths using the aforementioned definitions of high-symmetry points that have equally spaced k-points.

- Unless otherwise stated (for instance through parameters for functions), the appearance of the plot is dictated by the current matplotlib `rcParams`.

- A notable exception to the previous point `BandsData.setup_axes` for setting up the actual band structure axes.
  In general, you should call this function after initialising any matplotlib `Axes` object prior to any actual plotting tasks.

This package has been tested on Quantum ESPRESSO version >= 7.0. While it may work on older versions, there are no guarantees...

## License
Copyright (C) 2025 V. Ravindran
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
