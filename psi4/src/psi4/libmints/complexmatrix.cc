/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "psi4/libmints/complexmatrix.h"

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libmints/dimension.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

// For outfile in ComplexMatrix::print()
#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include <algorithm>
#include <cmath>

#ifdef USING_Einsums
#include <Einsums/TensorAlgebra.hpp>

// Raw LAPACK interface for ZHEEVR (Hermitian expert driver, MRRR algorithm),
// used by linalg::diagonalize() below. Declared locally instead of in
// libqt/lapack_intfc.h, which only wraps the real (D/S) LAPACK routines.
// When building with MKL (USING_LAPACK_MKL) plus OpenOrbitalOptimizer, the
// <mkl.h> pulled in by complexmatrix.h already declares the Fortran zheevr_
// (with MKL_Complex16* args), so only declare it ourselves otherwise.
#if !defined(USING_OpenOrbitalOptimizer) || !defined(USING_LAPACK_MKL)
extern "C" void zheevr_(char *jobz, char *range, char *uplo, int *n, std::complex<double> *a, int *lda, double *vl,
                        double *vu, int *il, int *iu, double *abstol, int *m, double *w, std::complex<double> *z,
                        int *ldz, int *isuppz, std::complex<double> *work, int *lwork, double *rwork, int *lrwork,
                        int *iwork, int *liwork, int *info);
#endif

// MKL_Complex16 (mkl_types.h, via <mkl.h>) is ABI-identical to
// std::complex<double>; adapt the complex pointers to the active declaration.
#if defined(USING_OpenOrbitalOptimizer) && defined(USING_LAPACK_MKL)
inline MKL_Complex16 *zheevr_cplx(std::complex<double> *p) { return reinterpret_cast<MKL_Complex16 *>(p); }
#else
inline std::complex<double> *zheevr_cplx(std::complex<double> *p) { return p; }
#endif
#endif

namespace psi {

#ifdef USING_Einsums

// Dereferencing a SharedMatrix requires Matrix to be fully qualified. To save
// compilation time, complexmatrix.h includes don't require matrix.h as well.
// Thus, SharedMatrix dereference is put here where matrix.h is included.
ComplexMatrix::ComplexMatrix(const std::shared_ptr<Matrix>& A)
    : ComplexMatrix(*A) {}

// Static helper for Matrix-valued constructor.
ComplexMatrix::TiledT ComplexMatrix::matrix_to_tiled_tensor(const Matrix& M) {
    const auto& rowspi = M.rowspi().blocks();
    const auto& colspi = M.colspi().blocks();
    const auto& name = M.name();
    TiledT T{name, rowspi, colspi};
    for (int h = 0; h < M.nirrep(); h++) {
        BlockT& Tb = T.tile(h, h);
        for (int i = 0; i < rowspi[h]; i++) {
            for (int j = 0; j < colspi[h]; j++) {
                Tb(i, j) = M.get(h, i, j);
            }
        }
    }
    return T;
}

// Public static method to create spin-blocked ComplexMatrix from Matrix.
// Each irrep becomes: A --> [[A, 0], [0, A]].as_type(complex)
SharedComplexMatrix ComplexMatrix::spin_blocked_from(const Matrix& A) {
    const int nirrep = A.nirrep();
    Dimension row_dim(nirrep);
    Dimension col_dim(nirrep);

    for (int h = 0; h < nirrep; h++) {
        row_dim[h] = A.rowspi(h) * 2;
        col_dim[h] = A.colspi(h) * 2;
    }

    auto B = std::make_shared<ComplexMatrix>(A.name(), row_dim, col_dim);
    B->zero();

    for (int h = 0; h < nirrep; h++) {
        const int r_ = A.rowspi(h);
        const int c_ = A.colspi(h);
        for (int i = 0; i < r_; i++) {
            for (int j = 0; j < c_; j++) {
				B->set(h, i, j, A(h, i, j));
				B->set(h, i + r_, j + c_, A(h, i, j));
            }
        }
    }

    return B;
}

SharedComplexMatrix ComplexMatrix::spin_blocked_from(const SharedMatrix& A) {
    return ComplexMatrix::spin_blocked_from(*A);
}

// Same as above, but you pass in the SharedComplexMatrix.
void ComplexMatrix::copy_matrix_to_spin_blocked(const Matrix& A, SharedComplexMatrix& B) {
    for (int h = 0; h < A.nirrep(); h++) {
        const int r_ = A.rowspi(h);
        const int c_ = A.colspi(h);
        for (int i = 0; i < r_; i++) {
            for (int j = 0; j < c_; j++) {
				B->set(h, i, j, A(h, i, j));
				B->set(h, i + r_, j + c_, A(h, i, j));
            }
        }
    }
}

#ifdef USING_OpenOrbitalOptimizer
arma::cx_mat ComplexMatrix::to_armadillo_matrix(int h) {
    int nc = coldim(h);
    int nr = rowdim(h);
    arma::cx_mat m(nr, nc);
    for (int ir = 0; ir < nr; ir++)
        for (int ic = 0; ic < nc; ic++) m(ir, ic) = get(h, ir, ic);
    return m;
}

void ComplexMatrix::from_armadillo_matrix(const arma::cx_mat& m, int h) {
    int nc = coldim(h);
    int nr = rowdim(h);
    for (int ir = 0; ir < nr; ir++)
        for (int ic = 0; ic < nc; ic++) set(h, ir, ic, m(ir, ic));
}
#endif

// self += alpha * other
//
// Implemented as a plain element loop (via operator(p, q), the same accessor used
// throughout cghf.cc/ComplexJK) rather than einsums::linear_algebra::axpy, whose
// generic TensorConcept dispatch appears to not work.
void ComplexMatrix::axpy(std::complex<double> alpha, const ComplexMatrix& other) {
    const auto& other_t = static_cast<const TiledT&>(other);
    if (tensor_.grid_size(0) != other_t.grid_size(0) || tensor_.grid_size(1) != other_t.grid_size(1)) {
        throw PSIEXCEPTION("ComplexMatrix::axpy: tile grids must match.");
    }
    for (int h = 0; h < static_cast<int>(other_t.grid_size(0)); ++h) {
        if (!other_t.has_tile(h, h) || other_t.has_zero_size(h, h)) continue;
        const auto& B = other_t.tile(h, h);
        auto& A = tensor_.tile(h, h);  // lazily allocated if missing
        const int nr = static_cast<int>(B.dim(0));
        const int nc = static_cast<int>(B.dim(1));
        for (int p = 0; p < nr; ++p) {
            for (int q = 0; q < nc; ++q) {
                A(p, q) += alpha * B(p, q);
            }
        }
    }
}

// Re(Tr(self^H other)), summed over diagonal tiles
double ComplexMatrix::vector_dot(const ComplexMatrix& other) const {
    const auto& other_t = other.tensor_;
    std::complex<double> total{0.0, 0.0};
    for (int h = 0; h < static_cast<int>(tensor_.grid_size(0)); ++h) {
        if (!tensor_.has_tile(h, h) || !other_t.has_tile(h, h)) continue;
        const auto& A = tensor_.tile(h, h);
        const auto& B = other_t.tile(h, h);
        const int nr = static_cast<int>(A.dim(0));
        const int nc = static_cast<int>(A.dim(1));
        for (int p = 0; p < nr; ++p) {
            for (int q = 0; q < nc; ++q) {
                total += std::conj(A(p, q)) * B(p, q);
            }
        }
    }
    return total.real();
}

std::complex<double> ComplexMatrix::product_trace(const ComplexMatrix& other) const {
    std::complex<double> E;

    using namespace einsums;
    tensor_algebra::einsum(Indices{}, &E, Indices{index::i, index::j}, tensor_, Indices{index::j, index::i},
                           other.tensor_);

    return E;
}

std::complex<double> ComplexMatrix::trace() const {
    std::complex<double> tr{0.0, 0.0};
    for (int h = 0; h < tensor_.grid_size(0); h++) {
        if (!tensor_.has_tile(h, h)) continue;
        const auto& tile = tensor_.tile(h, h);
        for (int i = 0; i < tensor_.tile_size(0)[h]; i++) {
            tr += tile(i, i);
        }
    }
    return tr;
}

double ComplexMatrix::rms() const {
    double sum = 0.0;
    long terms = 0;
    for (int h = 0; h < nirrep(); ++h) {
        const int nr = rowdim(h);
        const int nc = coldim(h);
        terms += static_cast<long>(nr) * nc;
        if (!tensor_.has_tile(h, h) || nr == 0 || nc == 0) continue;
        const auto& A = tensor_.tile(h, h);
        for (int i = 0; i < nr; ++i) {
            for (int j = 0; j < nc; ++j) {
                sum += std::norm(A(i, j));
            }
        }
    }
    if (terms == 0) return 0.0;
    return std::sqrt(sum / static_cast<double>(terms));
}

double ComplexMatrix::absmax() const {
    double max = 0.0;

    // Gets all filled tiles, unfilled are zeros
    for (const auto& [dims, tile] : tensor_.tiles()) {
        const auto& [nrow, ncol] = dims;
        for (int i = 0; i < nrow; ++i) {
            for (int j = 0; j < ncol; ++j) {
                max = std::max(max, std::abs(tile(i, j)));
            }
        }
    }

    return max;
}

// Raw per-tile complex sub-blocks to a PSIO file, mirroring Matrix::save with
// SaveType::SubBlocks (libmints/matrix.cc).
void ComplexMatrix::save(std::shared_ptr<PSIO>& psio, size_t fileno) {
    bool already_open = psio->open_check(fileno);
    if (!already_open) psio->open(fileno, PSIO_OPEN_OLD);

    for (int h = 0; h < static_cast<int>(tensor_.grid_size(0)); ++h) {
        if (!tensor_.has_tile(h, h) || tensor_.has_zero_size(h, h)) continue;
        auto& t = tensor_.tile(h, h);
        std::string entry = tensor_.name() + " Tile " + std::to_string(h);
#pragma message("Dangerous einsums tensor pointer operation used! A different solution is needed!")
        psio->write_entry(fileno, entry, (char*)t.data(), sizeof(std::complex<double>) * t.size());
    }

    if (!already_open) psio->close(fileno, 1);  // keep
}

// The ComplexMatrix must already have the right tile grid before loading (as with
// Matrix::load), e.g. constructed via ComplexMatrix(name, block_sizes).
void ComplexMatrix::load(std::shared_ptr<PSIO>& psio, size_t fileno) {
    bool already_open = psio->open_check(fileno);
    if (!already_open) psio->open(fileno, PSIO_OPEN_OLD);

    for (int h = 0; h < static_cast<int>(tensor_.grid_size(0)); ++h) {
        if (tensor_.has_zero_size(h, h)) continue;
        auto& t = tensor_.tile(h, h);  // lazily allocated to the declared size
        std::string entry = tensor_.name() + " Tile " + std::to_string(h);
#pragma message("Dangerous einsums tensor pointer operation used! A different solution is needed!")
        psio->read_entry(fileno, entry, (char*)t.data(), sizeof(std::complex<double>) * t.size());
    }

    if (!already_open) psio->close(fileno, 1);
}

void ComplexMatrix::print(std::string out, const char* extra) const {
    if (extra != nullptr) {
        throw PSIEXCEPTION("Not implemented");
    }

    std::shared_ptr<psi::PsiOutStream> printer = (out == "outfile" ? outfile : std::make_shared<PsiOutStream>(out));

    einsums::fprintln(*printer->stream(), tensor_);
}

SharedComplexMatrix ComplexMatrix::conjT() const {
    auto temp = std::make_shared<ComplexMatrix>(this->name() + "^H", this->colspi(), this->rowspi());

    // using keyT = const std::array<int, 2>;
    // using valueT = const einsums::Tensor<std::complex<double>, 2>;
    for (const auto& [key, value] : tensor_.tiles()) {
        const auto& [rowpi, colpi] = key;
        static_assert(std::is_same_v<decltype(rowpi), const int>);

        // necessarily: tensor_.has_tile(rowpi, colpi) == true
        // This adds a tile to temp.
        auto& ttile = temp->tensor_.tile(colpi, rowpi);

        // The template parameter <true> means take the conjugate.
#pragma message("Einsums::tensor_algebra::permute API will change as soon as v2.0")
        einsums::tensor_algebra::permute<true>(
            std::complex<double>{0.0}, einsums::Indices{einsums::index::i, einsums::index::j}, &ttile,
            std::complex<double>{1.0}, einsums::Indices{einsums::index::j, einsums::index::i}, value);
    }

    return temp;
}

namespace linalg {

namespace {

std::tuple<std::shared_ptr<Vector>, SharedComplexMatrix> diagonalize_in_place(ComplexMatrix& Forth) {
    Dimension nmopi_ = Forth.colspi();
    auto epsilon_ = std::make_shared<Vector>("Orbital energies", nmopi_);

    for (int h = 0; h < Forth.nirrep(); h++) {
        // Do not diagonalize 0x0 matrix
        if (nmopi_[h] == 0) continue;


        einsums::Tensor<double, 1> evals("Fock evals", nmopi_[h]);
        evals.zero();

        // C' = eig(F')
        // Hermitian eigensolver one block at a time. Raw LAPACK zheevr_
        // (expert driver, MRRR) rather than einsums::linear_algebra::heev,
        // which only wraps plain zheev: that routine is threaded under MKL but
        // serial in OpenBLAS/AOCL, so the raw expert driver gives a fairer
        // cross-vendor benchmark and reaches the backend's own code directly.
        auto &block = Forth.get(h);
        int n = nmopi_[h];
        int lda = static_cast<int>(block.stride(0));
        char jobz = 'V', range = 'A', uplo = 'U';
        double vl = 0.0, vu = 0.0;  // unused for RANGE='A'
        int il = 1, iu = n;         // unused for RANGE='A'
        double abstol = 0.0;        // 0 -> default (EPS * ||T||) tolerance
        int nevals = 0;             // out: number of eigenvalues found
        double *w = evals.data();
        std::vector<std::complex<double>> z(n * n);  // out: eigenvectors, column-major
        int ldz = n;
        std::vector<int> isuppz(2 * n);  // unreferenced for RANGE='A'; kept for safety
        int info = 0;

        // Workspace query (LWORK = LRWORK = LIWORK = -1 together), then the real
        // call. Querying keeps the buffer sizes portable across vendors.
        int lwork = -1, lrwork = -1, liwork = -1;
        std::complex<double> work_query[1];
        double rwork_query[1];
        int iwork_query[1];
        zheevr_(&jobz, &range, &uplo, &n, zheevr_cplx(block.data()), &lda, &vl, &vu, &il, &iu, &abstol, &nevals, w,
                zheevr_cplx(z.data()), &ldz, isuppz.data(), zheevr_cplx(work_query), &lwork, rwork_query, &lrwork,
                iwork_query, &liwork, &info);
        if (info != 0) throw PSIEXCEPTION("CGHF diagonalize: zheevr_ workspace query failed");
        lwork = static_cast<int>(work_query[0].real());
        lrwork = static_cast<int>(rwork_query[0]);
        liwork = iwork_query[0];

        std::vector<std::complex<double>> work(lwork);
        std::vector<double> rwork(lrwork);
        std::vector<int> iwork(liwork);
        zheevr_(&jobz, &range, &uplo, &n, zheevr_cplx(block.data()), &lda, &vl, &vu, &il, &iu, &abstol, &nevals, w,
                zheevr_cplx(z.data()), &ldz, isuppz.data(), zheevr_cplx(work.data()), &lwork, rwork.data(), &lrwork,
                iwork.data(), &liwork, &info);
        if (info != 0) throw PSIEXCEPTION("CGHF diagonalize: zheevr_ failed!");
        if (nevals != n) throw PSIEXCEPTION("CGHF diagonalize: zheevr_ returned the wrong number of eigenvalues");

        // zheevr_ writes the eigenvectors Q (of the column-major view V = block^T)
        // into z. LAPACK stored them column-major, so z[i + j*n] = Q[i][j], and the
        // row-major block must hold Q^T (block(i,j) = Q[j][i]) so that the conjT()
        // below yields the same eigenvector matrix as the einsums heev path.
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                block(i, j) = z[j + i * n];
            }
        }

        double last_value = - std::numeric_limits<double>::infinity();
        for (int m = 0; m < nmopi_[h]; m++) {
            const double& current_value = evals(m);
            if (last_value > current_value + 1e-16) throw PSIEXCEPTION("CGHF Orbitals are not ordered!");
            epsilon_->set(h, m, current_value);
            last_value = current_value;
        }
    }

    // heev retuns the wrong side, so we need to take the conjugate transpose for the proper eigenvectors
    SharedComplexMatrix temp = Forth.conjT();
    return std::make_tuple(epsilon_, temp);
}

}  // namespace

std::tuple<std::shared_ptr<Vector>, SharedComplexMatrix> diagonalize(const ComplexMatrix& F_) {
    ComplexMatrix Forth = F_; // copy as to not destroy original F_
    return diagonalize_in_place(Forth);
}

std::tuple<std::shared_ptr<Vector>, SharedComplexMatrix> diagonalize(
    const ComplexMatrix& F_, const ComplexMatrix& X_) {
    // Form F' = X'FX for canonical orthogonalization
    auto Forth = triplet(X_, F_, X_, true, false, false);
    Forth.set_name("Orthogonalized Fock");

    return diagonalize_in_place(Forth);
}

/**
 *  Previously, ``einsums::linalg::gemm<bool, bool>`` was used instead of this,
 *  However, linalg::gemm does the transpose **without conjugate** as of (v1.1.3).
 *  Substituting direct blas call until v2 upgrade. Don't worry, I have unit tests for it.
 *
 *  NOTE: when you move back to Einsums gemm, don't forget to explicitly specialize
 *  all the templates!
 */
namespace {

void doublet_into(const ComplexMatrix& A, const ComplexMatrix& B, bool AdjoinA, bool AdjoinB,
                  ComplexMatrix& C) {
    const auto& Atensor = static_cast<const ComplexMatrix::TiledT&>(A);
    const auto& Btensor = static_cast<const ComplexMatrix::TiledT&>(B);
    auto& Ctensor = static_cast<ComplexMatrix::TiledT&>(C);

    const int nirrep = A.nirrep();

    // ComplexMatrix stores only diagonal tiles (h,h).  For C = op(A) * op(B)
    // with diagonal-only storage this reduces to independent per-irrep gemms:
    //   C(h,h) = op(A(h,h)) * op(B(h,h))
    // Use einsums::blas::gemm directly so we can pass 'c' (conjugate-transpose)
    // instead of 't' (plain transpose) for the adjoint operations.
    for (int h = 0; h < nirrep; h++) {
        if (Atensor.has_zero_size(h, h) || Btensor.has_zero_size(h, h)) continue;
        if (!Atensor.has_tile(h, h) || !Btensor.has_tile(h, h)) continue;

        auto const &Atile = Atensor.tile(h, h);
        auto const &Btile = Btensor.tile(h, h);
        auto &Ctile = Ctensor.tile(h, h);

        using int_t = einsums::blas::int_t;

        const int_t m = static_cast<int_t>(AdjoinA ? Atile.dim(1) : Atile.dim(0));
        const int_t n = static_cast<int_t>(AdjoinB ? Btile.dim(0) : Btile.dim(1));
        const int_t k = static_cast<int_t>(AdjoinA ? Atile.dim(0) : Atile.dim(1));

        einsums::blas::gemm(AdjoinA ? 'c' : 'n', AdjoinB ? 'c' : 'n', m, n, k,
                            std::complex<double>{1.0}, Atile.data(), static_cast<int_t>(Atile.stride(0)),
                            Btile.data(), static_cast<int_t>(Btile.stride(0)),
                            std::complex<double>{0.0}, Ctile.data(), static_cast<int_t>(Ctile.stride(0)));
    }
}

}  // namespace

ComplexMatrix doublet(const ComplexMatrix& A, const ComplexMatrix& B, bool AdjoinA, bool AdjoinB) {
    const Dimension C_rowspi = (AdjoinA) ? A.colspi() : A.rowspi();
    const Dimension C_colspi = (AdjoinB) ? B.rowspi() : B.colspi();

    ComplexMatrix C{"T", C_rowspi, C_colspi};
    doublet_into(A, B, AdjoinA, AdjoinB, C);
    return C;
}

SharedComplexMatrix doublet(const SharedComplexMatrix& A, const SharedComplexMatrix& B,
                            bool AdjoinA, bool AdjoinB) {
    const Dimension C_rowspi = (AdjoinA) ? A->colspi() : A->rowspi();
    const Dimension C_colspi = (AdjoinB) ? B->rowspi() : B->colspi();
    auto C = std::make_shared<ComplexMatrix>("T", C_rowspi, C_colspi);
    doublet_into(*A, *B, AdjoinA, AdjoinB, *C);
    return C;
}

ComplexMatrix triplet(const ComplexMatrix& A, const ComplexMatrix& B, const ComplexMatrix& C,
                      bool AdjoinA, bool AdjoinB, bool AdjoinC) {
    auto BC = doublet(B, C, AdjoinB, AdjoinC);
    auto ABC = doublet(A, BC, AdjoinA, AdjoinB);
    return ABC;
}

SharedComplexMatrix triplet(const SharedComplexMatrix& A, const SharedComplexMatrix& B, const SharedComplexMatrix& C,
                            bool AdjoinA, bool AdjoinB, bool AdjoinC) {
    auto BC = doublet(B, C, AdjoinB, AdjoinC);
    return doublet(A, BC, AdjoinA, AdjoinB);
}

SharedComplexMatrix expm(const ComplexMatrix& A, std::complex<double> c) {
    // A == U.H @ np.diag(evals) @ U
    auto [evals, Uh] = diagonalize(A);
    const auto nirrep = A.nirrep();

    auto U = Uh->conjT();

    for (auto h = 0; h < nirrep; h++) {
        const auto dim = evals->dim(h);
        for (auto i = 0; i < dim; i++) {
            auto f = std::exp(c * evals->get(h, i));
            einsums::linear_algebra::scale_row(i, f, &U->get(h));
        }
    }

    return doublet(Uh, U);
}

}  // namespace linalg

#endif  // USING_Einsums

}  // namespace psi
