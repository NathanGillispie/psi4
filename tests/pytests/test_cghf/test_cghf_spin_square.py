"""
This test covers the spin_square method. It runs calculations for closed- and
open-shell molecules to determine if spin_square is being computed accurately.

core.CGHF.spin_square(self) returns a tuple of (⟨S²⟩, 2S+1), similar to PySCF.
"""
import pytest

import psi4
from addons import using

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, pytest.mark.cghf, *using("einsums")]

@pytest.fixture(autouse=True)
def clean_psi():
    """Provide a clean workspace for each test."""
    # pre-test cleanup
    psi4.core.clean()
    psi4.core.clean_options()

    yield

    # post-test cleanup
    psi4.core.clean()
    psi4.core.clean_options()


def test_cghf_spin_square_helium():
    """He should be a pure singlet: (0, 1)."""
    mol = psi4.geometry("""
    0 1
      He
    symmetry c1
    """)
    psi4.set_options({
        "basis": "sto-3g",
        "reference": "cghf",
        "scf_type": "direct",
        "df_scf_guess": False,
    })
    _, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)

    ss, multiplicity = wfn.spin_square()

    assert ss == pytest.approx(0.0, abs=1e-8)
    assert multiplicity == pytest.approx(1.0, abs=1e-8)


def test_cghf_spin_square_nitric_oxide():
    """NO should be about a doublet: (.75, 2), but it's contaminated."""
    mol = psi4.geometry("""
    0 2
      N
      O 1 1.165
    symmetry c1
    """)
    psi4.set_options({
        "basis": "6-31G",
        "reference": "cghf",
        "guess": "core",
        "scf_type": "direct",
        "df_scf_guess": False,
        "e_convergence": 5e-7
    })
    calc_e, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)

    assert calc_e == pytest.approx(-129.174262,abs=1e-6)

    ss, multiplicity = wfn.spin_square()

    assert ss == pytest.approx(0.932, abs=1e-2)
    assert multiplicity == pytest.approx(2.174, abs=1e-2)


def test_cghf_spin_square_nitrogen():
    """N should be about a quartet: (3.75, 4)."""
    mol = psi4.geometry("""
    0 4
      N
    symmetry c1
    """)
    psi4.set_options({
        "basis": "6-31G",
        "reference": "cghf",
        "guess": "core",
        "scf_type": "direct",
        "df_scf_guess": False,
        "e_convergence": 1e-8
    })
    calc_e, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)

    assert calc_e == pytest.approx(-54.38500771,abs=2e-8)

    ss, multiplicity = wfn.spin_square()

    assert ss == pytest.approx(3.75, abs=1e-2)
    assert multiplicity == pytest.approx(4, abs=1e-2)
