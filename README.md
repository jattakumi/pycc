PyCC
==============================
[//]: # (Badges)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Actions Build
Status](https://github.com/CrawfordGroup/pycc/workflows/CI/badge.svg)](https://github.com/CrawfordGroup/pycc/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/CrawfordGroup/pycc/branch/main/graph/badge.svg)](https://codecov.io/gh/CrawfordGroup/pycc/branch/main)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/lothian/pycc.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/lothian/pycc/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/lothian/pycc.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/lothian/pycc/context:python)

A Python-based coupled cluster implementation.  Current capabilities include:
  - Spin-adapted CCD, CC2, CCSD, CCSD(T), and CC3 energies
  - Triples-drivers for various approximate triples methods
  - RHF-CC2 and CCSD and densities
  - Real-time (RT) CC2 and CCSD
  - GPU implementations for multiple methods
  - Single- and mixed-precision arithmetic
  - Real-time CC2 and CCSD with a selection of integrators
  - PAO-, PNO-, and PNO++-CCSD energies RT-CC

Future plans:
  - RT-CC3
  - Linear and quadratic response functions
  - EOM-CC
  - Analytic gradients

This repository is currently under development. To do a developmental install, download this repository and type `pip install -e .` in the repository directory.

This package requires the following:
  - [psi4](https://psicode.org)
  - [numpy](https://numpy.org/)
  - [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)
  - [scipy](https://www.scipy.org/)
  - [pytorch](https://pytorch.org/)

### Copyright

Copyright (c) 2022, T. Daniel Crawford


#### Acknowledgements
 
Project structure based on the 
[MolSSI's](https://molssi.org) [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) Version 1.5.
