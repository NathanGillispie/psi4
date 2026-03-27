#!/usr/bin/env python
"""
Temporary stuff for amfX2C.
"""
import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

def test_amfx2c():
    """Dummy Test"""
    mol = psi4.core.Molecule.from_string("0 1\nH 0 0 0\nH .5 0 0")
    basis = psi4.core.BasisSet.build(mol, 'orbital', '6-31G*')

    amfx2c = psi4.core.amfx2c(mol, basis)

    amfx2c.compute()

    assert True

if __name__ == "__main__":
    test_amfx2c()


