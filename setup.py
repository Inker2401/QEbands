from setuptools import find_packages
from setuptools import setup, Extension

setup(name="QEbands",
      version="1.0.0",
      description="Visualise Quantum ESPRESSO band structures",
      packages=find_packages(),
      url='https://github.com/Inker2401/QEbands.git',
      author="Visagan Ravindran, Paul Dirk",
      license="GPLv3",
      python_requires='>=3.10',
      install_requires=[
          "spglib>=2.4.0",
          "numpy>=2.0.0",
          "matplotlib>=3.10.0",
          "ase>=3.18.1"],
      )
