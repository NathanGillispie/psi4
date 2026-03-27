#!/usr/bin/env python
"""
Temporary stuff for amfX2C.

TODO:
 - [ ] Test projecting to AO/SO orbitals
 - [ ] Add regular tests for grids (e.g. integrating (2π)^{-3/2}exp(-r^2 / 2) gives 1)
 - [ ] Create database of radial grids for C++ (mutable on python side?)
 - [ ] Eventually: build spherical grid *from* (r, f(r)) list? Is that even possible?
"""
import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

def test_amfx2c():
    """Dummy Test"""
    mol = psi4.core.Molecule.from_string("0 1\nH 0 0 0\nH .5 0 0")
    basis = psi4.core.BasisSet.build(mol, 'BASIS', '6-31G*')

    amfx2c = psi4.core.amfx2c(mol, basis) # test constructor
    bs_radii = psi4.core.amfx2c.bs_radii # test static readonly property

    amfx2c.compute()

    mints = psi4.core.MintsHelper(basis)
    S = mints.ao_overlap()
    print(S.to_array())

    assert True

if __name__ == "__main__":
    test_amfx2c()


