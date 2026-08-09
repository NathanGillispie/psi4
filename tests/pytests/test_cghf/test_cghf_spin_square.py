import pytest

import psi4
from addons import using

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, pytest.mark.cghf, *using("einsums")]


def test_cghf_spin_square_helium():
    """Closed-shell He CGHF should be a pure singlet: ⟨S²⟩ ≈ 0, 2S+1 ≈ 1."""
    mol = psi4.geometry("""
    0 1
    He
    symmetry c1
    """)
    psi4.set_options({
        "basis": "sto-3g",
        "reference": "cghf",
        "guess": "core",
        "scf_type": "direct",
        "df_scf_guess": False,
        "diis": False,
        "scf_initial_accelerator": "none",
    })
    _, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)

    result = wfn.spin_square()
    assert isinstance(result, tuple)
    assert len(result) == 2
    ss, multiplicity = result

    assert ss == pytest.approx(0.0, abs=1e-8)
    assert multiplicity == pytest.approx(1.0, abs=1e-8)
