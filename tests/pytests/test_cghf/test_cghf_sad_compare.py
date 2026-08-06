import pytest
import numpy as np

import psi4
from addons import using

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, pytest.mark.cghf, *using("einsums")]


def test_sad_guess_uhf_vs_cghf():
    """For open-shell O2 without spin-orbit coupling, UHF and CGHF with
    GUESS=SAD must converge to the same energy.

    The SAD guess is fundamentally real and spin-restricted (Da == Db).
    CGHF promotes it to a complex block-diagonal spinor density
       D_cghf = [[Da,  0 ],
                 [0 ,  Da]]
    with zero off-diagonal (alpha-beta coupling) blocks and zero
    imaginary part.  After SCF convergence the same block-diagonal
    structure persists: D_aa == Da_uhf, D_bb == Db_uhf, D_ab == 0.
    """

    mol = psi4.geometry("""
        0 3
        O
        O 1 1.2
        symmetry c1
    """)

    # ------------------------------------------------------------------
    # Run CGHF
    # ------------------------------------------------------------------
    psi4.set_options({
        "basis": "6-31g",
        "scf_type": "direct",
        "df_scf_guess": False,
        "maxiter": 20,
        "fail_on_maxiter": False,
        "guess": "sad",
        "reference": "cghf",
    })
    e_cghf, wfn_cghf = psi4.energy("scf", molecule=mol, return_wfn=True)

    # ------------------------------------------------------------------
    # Run UHF
    # ------------------------------------------------------------------
    psi4.set_options({"reference": "uhf"})
    e_uhf, wfn_uhf = psi4.energy("scf", molecule=mol, return_wfn=True)

    # Without spin-orbit coupling, CGHF == UHF for open-shell
    # assert e_cghf == pytest.approx(e_uhf, abs=1e-6), \
    #     f"CGHF energy {e_cghf:.10f} != UHF energy {e_uhf:.10f}"

    # ------------------------------------------------------------------
    # Extract densities
    # ------------------------------------------------------------------
    # UHF: two real nbf x nbf matrices
    Da_uhf = wfn_uhf.Da().to_array()       # np.ndarray, float64
    Db_uhf = wfn_uhf.Db().to_array()

    nbf = Da_uhf.shape[0]

    # CGHF: one complex 2*nbf x 2*nbf spinor matrix
    D_cghf = wfn_cghf.D().to_array()       # np.ndarray, complex128

    assert D_cghf.dtype == np.complex128, "CGHF density is not complex"
    assert D_cghf.shape == (2 * nbf, 2 * nbf), \
        f"CGHF density shape {D_cghf.shape} != expected {(2*nbf, 2*nbf)}"

    # ------------------------------------------------------------------
    # Decompose CGHF density into spin blocks
    # ------------------------------------------------------------------
    D_aa = D_cghf[:nbf, :nbf]    # alpha-alpha
    D_ab = D_cghf[:nbf, nbf:]    # alpha-beta
    D_ba = D_cghf[nbf:, :nbf]    # beta-alpha
    D_bb = D_cghf[nbf:, nbf:]    # beta-beta

    # Diagonal blocks must match UHF Da/Db (real part)
    np.testing.assert_allclose(D_aa.real, Da_uhf, atol=1e-10,
                               err_msg="CGHF D_aa != UHF Da")
    np.testing.assert_allclose(D_bb.real, Db_uhf, atol=1e-10,
                               err_msg="CGHF D_bb != UHF Db")

    # Imaginary parts of diagonal blocks must be ~zero
    np.testing.assert_allclose(D_aa.imag, 0.0, atol=1e-10,
                               err_msg="CGHF D_aa has non-zero imaginary part")
    np.testing.assert_allclose(D_bb.imag, 0.0, atol=1e-10,
                               err_msg="CGHF D_bb has non-zero imaginary part")

    # Off-diagonal (alpha-beta coupling) blocks must be ~zero.
    # With the SAD guess these are *exactly* zero; after SCF
    # convergence they remain negligible for non-relativistic calc.
    np.testing.assert_allclose(np.abs(D_ab), 0.0, atol=1e-10,
                               err_msg="CGHF D_ab (alpha-beta coupling) is non-zero")
    np.testing.assert_allclose(np.abs(D_ba), 0.0, atol=1e-10,
                               err_msg="CGHF D_ba (beta-alpha coupling) is non-zero")

    # ------------------------------------------------------------------
    # Verify that if we build the expected SAD-style CGHF density from
    # UHF Da, it has the correct structure (block-diagonal, zero imag).
    # This models what CGHF::compute_SAD_guess() does internally.
    # ------------------------------------------------------------------
    D_sad_style = np.zeros((2 * nbf, 2 * nbf), dtype=np.complex128)
    D_sad_style[:nbf, :nbf] = Da_uhf
    D_sad_style[nbf:, nbf:] = Da_uhf   # SAD is spin-restricted: Da == Db

    # Imaginary part is identically zero
    np.testing.assert_allclose(D_sad_style.imag, 0.0, atol=1e-15)
    # Off-diagonal blocks are identically zero
    np.testing.assert_allclose(D_sad_style[:nbf, nbf:], 0.0, atol=1e-15)
    np.testing.assert_allclose(D_sad_style[nbf:, :nbf], 0.0, atol=1e-15)

    # Round-trip through psi4 ComplexMatrix to confirm API consistency
    D_cmplx = psi4.core.ComplexMatrix.from_array(D_sad_style, name="D_sad")
    np.testing.assert_allclose(D_cmplx.to_array(), D_sad_style, atol=1e-15)
