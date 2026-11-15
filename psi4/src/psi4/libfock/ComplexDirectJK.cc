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

#include "psi4/libfock/ComplexJK.h"

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/liboptions/liboptions.h"

#include "psi4/libmints/twobody.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libqt/qt.h"

#include <algorithm>
#include <array>
#include <complex>
#include <string>
#include <span>
#include <utility>
#include <vector>

namespace psi {

ComplexDirectJK::ComplexDirectJK(std::shared_ptr<BasisSet> primary, Options& options)
    : ComplexJK(primary), options_(options) {
    common_init();
}

ComplexDirectJK::~ComplexDirectJK() {}

void ComplexDirectJK::common_init() {
    if (options_.get_int("INCFOCK_FULL_FOCK_EVERY") <= 0) {
        throw PSIEXCEPTION("Invalid input for option INCFOCK_FULL_FOCK_EVERY (<= 0)");
    }

    auto screening_type = options_.get_str("SCREENING");
    // if (screening_type == "DENSITY") {
    //     throw PSIEXCEPTION("ComplexDirectJK does not support SCREENING=DENSITY yet.");
    // }
    do_csam_ = (screening_type == "CSAM");
    computed_shells_per_iter_["Quartets"] = {};

    // Set up AO2USO transform 
    ComplexJK::common_init();
}

size_t ComplexDirectJK::num_computed_shells() { return num_computed_shells_; }

size_t ComplexDirectJK::memory_estimate() {
    return 0; // Effectively?
}

void ComplexDirectJK::preiterations() { /*no-op*/ }

void ComplexDirectJK::compute_JK() {
    if (!do_J_ && !do_K_) {
        outfile->Printf("\n  WARNING: ComplexDirectJK tried to compute JK, but "
                        "do_J_ and do_K_ were both set to false.\n");
        return;
    }

    num_computed_shells_ = 0L;

    auto factory = std::make_shared<IntegralFactory>(primary_, primary_, primary_, primary_);
    const int nbf = primary_->nbf();

    auto ints = std::shared_ptr<TwoBodyAOInt>(factory->eri());
    if (options_.get_str("SCREENING") == "DENSITY") ints->update_density_complex(D_ao_);

    for (size_t N = 0; N < D_.size(); N++) {
        if (!(do_J_ && do_K_)) {
            // TODO: figure out later
            throw PSIEXCEPTION("Both J and K must be computed with ComplexJK and SCF_TYPE DIRECT");
        }

        const auto& D_ref = D_ao_[N]->get(0);
        const int dim = D_ref.dim(0);
        auto& J_out = J_ao_[N]->get(0);
        auto& K_out = K_ao_[N]->get(0);

	// Added in additional flag lr_symmetic_ for case of non-hermitian density analagous to DirectJK.cc
	// build_JK_matrices templated as <FillJ, FillK, SpinBlocked, Hermitian>
        const bool hermitian = lr_symmetric_;

        if (dim == nbf) {
            // Plain (non spin-blocked) complex density
            if (hermitian)
                build_JK_matrices<true, true, false, true>(ints, D_ref, &J_out, &K_out);
            else
                build_JK_matrices<true, true, false, false>(ints, D_ref, &J_out, &K_out);
        } else if (dim == 2 * nbf) {
            // Generalized (CGHF) spin-blocked density. D and J/K are 2x2 block
            // matrices of nbf x nbf blocks: [[D_aa, D_ab], [D_ba, D_bb]].
            // A single ERI pass contracts all four spin blocks (and the summed
            // Coulomb density) so the integrals are only computed once.
            if (hermitian)
                build_JK_matrices<true, true, true, true>(ints, D_ref, &J_out, &K_out);
            else
                build_JK_matrices<true, true, true, false>(ints, D_ref, &J_out, &K_out);
        } else {
            throw PSIEXCEPTION("ComplexDirectJK: density block dimension (" + std::to_string(dim) +
                               ") must equal the number of AO basis functions (" + std::to_string(nbf) +
                               ") or twice that (GHF spin-blocked density).");
        }
    }

    if (get_bench()) {
        computed_shells_per_iter_["Quartets"].push_back(num_computed_shells());
    }
}

namespace {

using ComplexT = einsums::Tensor<std::complex<double>, 2>;

// Minimal integer range [first, last) used in range-for loops. This replaces
// std::views::iota, which fails to compile under clang with libstdc++ < 13.
// Whenever Psi4's minimum GCC version is bumped to at least 14, we should
// revert back to using std::views::iota.
struct integer_range {
    std::size_t first;
    std::size_t last;

    struct iterator {
        std::size_t value;
        std::size_t operator*() const { return value; }
        iterator& operator++() { ++value; return *this; }
        bool operator==(const iterator& other) const { return value == other.value; }
        bool operator!=(const iterator& other) const { return value != other.value; }
    };

    iterator begin() const { return {first}; }
    iterator end() const { return {last}; }
};

// Helper method to remove the indexing madness. Produces iterable list of iterators.
// Converts Something like [0,1,4,5] into [[0, 1), [1, 4), [4, 5)].
auto partition(std::span<const std::size_t> atom_to_shell) {
    std::vector<integer_range> result;
    result.reserve(atom_to_shell.size());
    for (std::size_t atom = 0; atom + 1 < atom_to_shell.size(); ++atom)
        result.push_back(integer_range{atom_to_shell[atom], atom_to_shell[atom + 1]});
    return result;
}

// Like above but enumerated. Converts Something like [0,1,4,5] into
//   [{0, [0, 1)}, {1, [1, 4)}, {2, [4, 5)}].
auto partition_with_idx(std::span<const std::size_t> atom_to_shell) {
    std::vector<std::pair<std::size_t, integer_range>> result;
    result.reserve(atom_to_shell.size());
    for (std::size_t atom = 0; atom + 1 < atom_to_shell.size(); ++atom)
        result.emplace_back(atom, integer_range{atom_to_shell[atom], atom_to_shell[atom + 1]});
    return result;
}

// Contiguous AO range [first, first + size) of all functions on one center.
struct CenterBlock {
    std::size_t first;
    std::size_t size;
};

// Basis function indices of a single integral (pq|rs).
struct QuartetIndex {
    std::size_t p, q, r, s;
};

// For populating other triangular half of real J matrix (A <- (A + A^T) / 2)
void symmetrize(ComplexT& A, std::size_t n) {
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < i; j++) {
            const auto avg = 0.5 * (A(i, j) + A(j, i));
            A(i, j) = avg;
            A(j, i) = avg;
        }
    }
}

// For populating other triangular half of complex K matrix (A <- (A + A^H) / 2)
void hermitize(ComplexT& A, std::size_t n) {
    for (std::size_t i = 0; i < n; i++) {
        A(i, i) = std::real(A(i, i));
        for (std::size_t j = 0; j < i; j++) {
            const auto avg = 0.5 * (A(i, j) + std::conj(A(j, i)));
            A(i, j) = avg;
            A(j, i) = std::conj(avg);
        }
    }
}

// Pointer to row i of a block. All scratch data of the J/K build lives in
// ComplexMatrix objects, whose blocks are Einsums tensors. A row is therefore
// a contiguous run that the innermost loops walk with a raw pointer.
// Necessary for the compiler to vectorize (SIMD).
inline std::complex<double>* row_ptr(ComplexT& T, std::size_t i) { return T.data() + i * T.stride(0); }
inline const std::complex<double>* row_ptr(const ComplexT& T, std::size_t i) {
    return T.data() + i * T.stride(0);
}

// View a run of complex numbers as (re, im) pairs of doubles. std::complex
// guarantees this layout, and the standard permits the access.
inline double* as_reals(std::complex<double>* z) { return reinterpret_cast<double*>(z); }
inline const double* as_reals(const std::complex<double>* z) { return reinterpret_cast<const double*>(z); }

// Zero the leading nrow x ncol block.
void zero_block(ComplexT& T, std::size_t nrow, std::size_t ncol) {
    for (std::size_t i = 0; i < nrow; i++) std::fill_n(row_ptr(T, i), ncol, std::complex<double>{});
}

// Add the leading block of an atom-local accumulator into its (row, col)
// center block of a global matrix.
void stripe_block(ComplexT& dst, const ComplexT& src, CenterBlock row, CenterBlock col) {
    for (std::size_t i = 0; i < row.size; i++) {
        const std::complex<double>* src_row = row_ptr(src, i);
        for (std::size_t j = 0; j < col.size; j++) {
            row_ptr(dst, row.first + i)[col.first + j] += src_row[j];
        }
    }
}

// One exchange stream: a density X and the spin block of K it contributes to.
// A stream fills four atom-local buffers, one per unique-quartet:
//
//    K_pr += D_qs (pq|rs)
//    K_ps += D_qr (pq|rs)
//    K_qr += D_ps (pq|rs)
//    K_qs += D_pr (pq|rs)
//
// The other four terms that pop up from the 8-fold symmetry are:
//
//    K_rp += D_sq (pq|rs)
//    K_sp += D_rq (pq|rs)
//    K_rq += D_sp (pq|rs)
//    K_sq += D_rp (pq|rs)
//
// For a Hermitian D this second set of terms are just the conjugate transposes of
// the first set. Only the first four unique-quartets are evaluated and the full
// K contraction is recovered at the end by K <- (K + K^H)/2.
struct ExchangeStream {
    /// Position of the density block X inside D, and of its K block inside K
    std::size_t row_off;
    std::size_t col_off;
    /// Contract X^T and accumulate K^T instead
    bool transposed;
};

// Atom-local exchange accumulators for NStream exchange streams.
// Every stream shares the same (real) eri, so the update of all streams
// is a single short vector operation on those doubles.
template <std::size_t NStream>
struct ExchangeBuffers {
    /// Doubles per matrix element: (re, im) of every stream
    static constexpr std::size_t width = 2 * NStream;

    std::array<ExchangeStream, NStream> streams;
    /// Interleaved densities, nbf x (nbf * NStream), global AO indices
    ComplexMatrix X_matrix;
    /// Interleaved accumulators, each max_nfuncs x (max_nfuncs * NStream)
    std::array<ComplexMatrix, 4> KT_matrix;
    /// The (single) blocks of the matrices above
    ComplexT* X;
    std::array<ComplexT*, 4> KT;

    ExchangeBuffers(const ComplexT& D, std::size_t nbf, std::size_t max_nfuncs,
                    const std::array<ExchangeStream, NStream>& exchange_streams)
        : streams(exchange_streams) {
        const std::size_t ncopy = std::max<std::size_t>(NStream, 1);
        X_matrix = ComplexMatrix("X", static_cast<int>(nbf), static_cast<int>(nbf * ncopy));
        X = &X_matrix.get(0);
        for (std::size_t n = 0; n < 4; n++) {
            KT_matrix[n] = ComplexMatrix("KT", static_cast<int>(max_nfuncs), static_cast<int>(max_nfuncs * ncopy));
            KT[n] = &KT_matrix[n].get(0);
        }
        for (std::size_t h = 0; h < NStream; h++) {
            const auto& stream = streams[h];
            for (std::size_t a = 0; a < nbf; a++) {
                std::complex<double>* X_row = row_ptr(*X, a);
                for (std::size_t b = 0; b < nbf; b++) {
                    X_row[b * NStream + h] = stream.transposed ? row_ptr(D, b + stream.row_off)[a + stream.col_off]
                                                               : row_ptr(D, a + stream.row_off)[b + stream.col_off];
                }
            }
        }
    }

    ExchangeBuffers(const ExchangeBuffers&) = delete;
    ExchangeBuffers& operator=(const ExchangeBuffers&) = delete;

    static std::array<std::pair<CenterBlock, CenterBlock>, 4> layout(CenterBlock P, CenterBlock Q, CenterBlock R,
                                                                    CenterBlock S) {
        return {{{P, R}, {P, S}, {Q, R}, {Q, S}}};
    }

    void zero(CenterBlock P, CenterBlock Q, CenterBlock R, CenterBlock S) {
        const auto blocks = layout(P, Q, R, S);
        for (std::size_t n = 0; n < 4; n++) zero_block(*KT[n], blocks[n].first.size, blocks[n].second.size * NStream);
    }

    // The accumulation of all terms that depend on eri (pq|rs) in a vectorized form for efficient SIMD compilation.
    inline void accumulate_row(const QuartetIndex& lo, const QuartetIndex& ao, std::size_t ns, const double* I,
                               double scale) {
        const double* __restrict__ X_qs = as_reals(row_ptr(*X, ao.q) + ao.s * NStream);
        const double* __restrict__ X_ps = as_reals(row_ptr(*X, ao.p) + ao.s * NStream);
        const double* __restrict__ X_qr = as_reals(row_ptr(*X, ao.q) + ao.r * NStream);
        const double* __restrict__ X_pr = as_reals(row_ptr(*X, ao.p) + ao.r * NStream);
        double* __restrict__ K_ps = as_reals(row_ptr(*KT[1], lo.p) + lo.s * NStream);
        double* __restrict__ K_qs = as_reals(row_ptr(*KT[3], lo.q) + lo.s * NStream);

        std::array<double, width> K_pr{};
        std::array<double, width> K_qr{};
        for (std::size_t s = 0; s < ns; s++) {
            const double value = scale * I[s];
#pragma omp simd
            for (std::size_t c = 0; c < width; c++) {
                K_pr[c] += X_qs[s * width + c] * value;
                K_qr[c] += X_ps[s * width + c] * value;
                K_ps[s * width + c] += X_qr[c] * value;
                K_qs[s * width + c] += X_pr[c] * value;
            }
        }

        double* __restrict__ K_pr_out = as_reals(row_ptr(*KT[0], lo.p) + lo.r * NStream);
        double* __restrict__ K_qr_out = as_reals(row_ptr(*KT[2], lo.q) + lo.r * NStream);
        for (std::size_t c = 0; c < width; c++) {
            K_pr_out[c] += K_pr[c];
            K_qr_out[c] += K_qr[c];
        }
    }

    void stripe(ComplexT& K, CenterBlock P, CenterBlock Q, CenterBlock R, CenterBlock S) const {
        const auto blocks = layout(P, Q, R, S);
        for (std::size_t n = 0; n < 4; n++) {
            const CenterBlock row = blocks[n].first;
            const CenterBlock col = blocks[n].second;
            for (std::size_t i = 0; i < row.size; i++) {
                const std::complex<double>* KT_row = row_ptr(*KT[n], i);
                for (std::size_t j = 0; j < col.size; j++) {
                    for (std::size_t h = 0; h < NStream; h++) {
                        const auto& stream = streams[h];
                        const std::complex<double> value = KT_row[j * NStream + h];
                        if (stream.transposed)
                            row_ptr(K, col.first + j + stream.row_off)[row.first + i + stream.col_off] += value;
                        else
                            row_ptr(K, row.first + i + stream.row_off)[col.first + j + stream.col_off] += value;
                    }
                }
            }
        }
    }
};

}

// The template allows us to provide the most optimized method at compile time
// using constexpr. No longer using scratch matrices. Benchmarking shows this
// being only marginally faster (<5%), but the code smells less.
//
// SpinBlocked selects between a plain complex density (nbf x nbf) and a
// generalized (CGHF) spin-blocked density (2 nbf x 2 nbf). It is a compile-time
// switch: both paths share the atom/shell-blocked skeleton and differ only in
// how the density is decomposed and how J/K are accumulated and striped back,
// with no runtime branching inside the hot loops.
template<bool FillJ, bool FillK, bool SpinBlocked, bool Hermitian>
void ComplexDirectJK::build_JK_matrices(std::shared_ptr<TwoBodyAOInt> ints, const ComplexT& D, ComplexT* J, ComplexT* K) {
    timer_on("build_JK_matrices");

    const size_t nbf = primary_->nbf();

    // J contraction only depends on D_J = D_c + D_c^T where D_c = D_aa + D_bb.
    const int nJ = FillJ ? static_cast<int>(nbf) : 1;
    ComplexMatrix D_J_matrix("D_J", nJ, nJ);
    ComplexT& D_J = D_J_matrix.get(0);
    if constexpr (FillJ) {
        auto D_c = [&](size_t p, size_t q) -> std::complex<double> {
            if constexpr (SpinBlocked)
                return D(p, q) + D(p + nbf, q + nbf);
            else
                return D(p, q);
        };
        for (size_t p = 0; p < nbf; p++) {
            for (size_t q = 0; q < nbf; q++) {
                const std::complex<double> sym = D_c(p, q) + D_c(q, p);
                if constexpr (Hermitian)
                    row_ptr(D_J, p)[q] = std::real(sym);
                else
                    row_ptr(D_J, p)[q] = sym;
            }
        }
    }

    if constexpr(FillJ) J->zero();
    if constexpr(FillK) K->zero();

    // => Atomic Task Blocking <= //
    // One task = all shells on one center. Shells stay in basis order (identity map).

    // shell index that starts each center
    std::vector<size_t> atom_to_shell;
    // boundaries of each partition
    std::vector<size_t> shell_to_basis;

    /*******************************************************
     * Consider H2O with STO-3G...                         *
     *                                                     *
     *   3 atoms:       H  ,         O           ,   H     *
     *   5 shells:    [ S ], [ S , S ,    P     ], [ S ]   *
     *   7 functions: [[s]], [[s],[s],[px,py,pz]], [[s]]   *
     *                                                     *
     * atom_to_shell  [ 0,           1,              4, 5] *
     * shell_to_basis [ 0,     1,  2,     3,         6, 7] *
     *                                                     *
     *******************************************************/

    // => Helper functions that need internal variables <= //

    auto nfunctions_in_shell = [this](const size_t& shell_idx) {
        return this->primary_->shell(shell_idx).nfunction();
    };

    auto function_index_of_shell = [this](const size_t& shell_idx) {
        return this->primary_->shell(shell_idx).function_index();
    };

    // Returns the basis function index that begins shell ``shell_idx`` where 0
    // indicates the first basis function of the first shell of atom ``task``.
    auto basis_index_of_shell_from_atom = [&atom_to_shell, &shell_to_basis](const size_t& shell_idx, const size_t& task) {
        return shell_to_basis[shell_idx] - shell_to_basis[atom_to_shell[task]];
    };

    // Returns a view of shell indices for given center idx.
    auto shells_on_center = [&atom_to_shell](size_t task) {
        return integer_range{atom_to_shell[task], atom_to_shell[task + 1]};
    };

    // Returns the contiguous AO range of all functions on center ``task``.
    auto functions_on_center = [&atom_to_shell, &shell_to_basis](size_t task) {
        const size_t first = shell_to_basis[atom_to_shell[task]];
        return CenterBlock{first, shell_to_basis[atom_to_shell[task + 1]] - first};
    };

    // => Welcome to the jungle <=

    // Total number of shells for all centers
    const size_t nshell = primary_->nshell();
    {
        int atomic_ind = -1;

        size_t total_nfuncs = 0;
        shell_to_basis.push_back(0);
        for (size_t P = 0; P < nshell; P++) {
            const auto& shell = primary_->shell(P);

            total_nfuncs += shell.nfunction();
            shell_to_basis.push_back(total_nfuncs);

            if (shell.ncenter() > atomic_ind) {
                atom_to_shell.push_back(P);
                atomic_ind++;
            }
        }
        atom_to_shell.push_back(nshell);
    }

    // Largest number of functions for any center. Used to allocate temporary
    // matrix for J and K.
    size_t max_nfuncs_per_center = 0;
    for (auto centers : partition(atom_to_shell)) {
        size_t size = 0;
        for (size_t shell : centers) {
            size += primary_->shell(shell).nfunction();
        }
        max_nfuncs_per_center = std::max(max_nfuncs_per_center, size);
    }

    // Local AO blocks for the current atom quartet. Only the leading
    // n_A x n_B block of each is used for a given quartet of centers.
    // Coulomb accumulators J_PQ = (J)_1^+ and J_RS = (J)_2^+.
    const int nJT = FillJ ? static_cast<int>(max_nfuncs_per_center) : 1;
    ComplexMatrix J_PQ_matrix("J_PQ", nJT, nJT);
    ComplexMatrix J_RS_matrix("J_RS", nJT, nJT);
    ComplexT& J_PQ = J_PQ_matrix.get(0);
    ComplexT& J_RS = J_RS_matrix.get(0);

    // Exchange streams, per density block (see ExchangeStream):
    //   plain density:        D                (1 stream if Hermitian, else 2)
    //   spin-blocked density: D_aa, D_bb       (1 stream each if Hermitian, else 2)
    //                         D_ab             (2 streams)
    //                         D_ba             (2 streams, non-Hermitian only)
    // The blocks of a generalized density D = [[D_aa, D_ab], [D_ba, D_bb]] are
    // addressed by their offsets; for a Hermitian one, K_ba = K_ab^H.
    constexpr size_t NStream = !FillK ? 0 : (SpinBlocked ? (Hermitian ? 4 : 8) : (Hermitian ? 1 : 2));
    const auto exchange_streams = [&]() -> std::array<ExchangeStream, NStream> {
        if constexpr (!FillK) {
            return {};
        } else if constexpr (!SpinBlocked && Hermitian) {
            return {{{0, 0, false}}};
        } else if constexpr (!SpinBlocked) {
            return {{{0, 0, false}, {0, 0, true}}};
        } else if constexpr (Hermitian) {
            return {{{0, 0, false}, {nbf, nbf, false}, {0, nbf, false}, {0, nbf, true}}};
        } else {
            return {{{0, 0, false}, {0, 0, true}, {nbf, nbf, false}, {nbf, nbf, true},
                     {0, nbf, false}, {0, nbf, true}, {nbf, 0, false}, {nbf, 0, true}}};
        }
    }();
    [[maybe_unused]] ExchangeBuffers<NStream> exchange(D, FillK ? nbf : 1, FillK ? max_nfuncs_per_center : 1,
                                                       exchange_streams);

    // Significant atom/task pairs (PQ|-- with Qatom <= Patom
    std::vector<std::pair<size_t, size_t>> significant_pairs;
    for (auto [Patom, Pshells] : partition_with_idx(atom_to_shell)) {
        for (auto [Qatom, Qshells] : partition_with_idx(atom_to_shell)) {
            if (Qatom > Patom) continue;
            bool found_significant_pair = false;
            for (size_t Ps : Pshells) {
                for (size_t Qs : Qshells) {
                    if (ints->shell_pair_significant(Ps, Qs)) {
                        found_significant_pair = true;
                        significant_pairs.emplace_back(Patom, Qatom);
                        break;
                    }
                }
                if (found_significant_pair) break;
            }
        }
    }

    size_t computed_shells = 0L;
    for (const auto [Patom, Qatom] : significant_pairs) {
        for (const auto [Ratom, Satom] : significant_pairs) {
            if (Ratom > Patom) continue;

            const CenterBlock Pfuncs = functions_on_center(Patom);
            const CenterBlock Qfuncs = functions_on_center(Qatom);
            const CenterBlock Rfuncs = functions_on_center(Ratom);
            const CenterBlock Sfuncs = functions_on_center(Satom);

            bool touched = false;

            // Process the unique shells: P >= Q, R >= S, PQ >= RS
            for (auto Ps : shells_on_center(Patom) ) {
                for (auto Qs : shells_on_center(Qatom) ) {
                    if (Qs > Ps) continue;
                    if (!ints->shell_pair_significant(Ps, Qs)) continue;
                    for (auto Rs : shells_on_center(Ratom)) {
                        for (auto Ss : shells_on_center(Satom)) {
                            if (Ss > Rs) continue;
                            if (Rs * nshell + Ss > Ps * nshell + Qs) continue;
                            if (!ints->shell_pair_significant(Rs, Ss)) continue;
                            if (!ints->shell_significant(Ps, Qs, Rs, Ss)) continue;

                            // Compute ERI
                            if (ints->compute_shell(Ps, Qs, Rs, Ss) == 0) continue;
                            const double* buf = ints->buffer();

                            // Don't forget to like, share, subscribe and compute that shell!
                            computed_shells++;

                            // Zeroing only the blocks this quartet of centers uses
                            if (!touched) {
                                if constexpr (FillJ) {
                                    zero_block(J_PQ, Pfuncs.size, Qfuncs.size);
                                    zero_block(J_RS, Rfuncs.size, Sfuncs.size);
                                }
                                if constexpr (FillK) {
                                    exchange.zero(Pfuncs, Qfuncs, Rfuncs, Sfuncs);
                                }
                                touched = true;
                            }

                            const size_t Psize = nfunctions_in_shell(Ps);
                            const size_t Qsize = nfunctions_in_shell(Qs);
                            const size_t Rsize = nfunctions_in_shell(Rs);
                            const size_t Ssize = nfunctions_in_shell(Ss);

                            // Global basis function index. 0 means the first function
                            // of the first shell of the first atom. aka AO index.
                            const size_t Pao = function_index_of_shell(Ps);
                            const size_t Qao = function_index_of_shell(Qs);
                            const size_t Rao = function_index_of_shell(Rs);
                            const size_t Sao = function_index_of_shell(Ss);

                            // basis index starting shell Ps from atom Patom.
                            // Let's call these "lo" for local orbital. Like
                            // AO but where 0 means the first AO of the given atom.
                            const size_t Plo = basis_index_of_shell_from_atom(Ps, Patom);
                            const size_t Qlo = basis_index_of_shell_from_atom(Qs, Qatom);
                            const size_t Rlo = basis_index_of_shell_from_atom(Rs, Ratom);
                            const size_t Slo = basis_index_of_shell_from_atom(Ss, Satom);

                            // Degeneracy of the unique quartet: 1/2 for each of P == Q,
                            // R == S and PQ == RS, since such a shell quartet already
                            // contains its own permutations.
                            double prefactor = 1.0;
                            if (Ps == Qs) prefactor *= 0.5;
                            if (Rs == Ss) prefactor *= 0.5;
                            if (Ps == Rs && Qs == Ss) prefactor *= 0.5;
                            // Doubled where the final (J + J^T)/2 or (K + K^H)/2 averages
                            // in the half that is never accumulated (DirectJK's
                            // post-hoc scale(2.0), folded into the prefactor).
                            const double J_prefactor = 2.0 * prefactor;
                            const double K_prefactor = (Hermitian ? 2.0 : 1.0) * prefactor;

                            // The integrals are stored with s running fastest, so one
                            // (p, q, r) owns a contiguous row of Ssize of them, and so
                            // do the density and accumulator rows they are contracted
                            // with.
                            for (size_t p = 0; p < Psize; p++) {
                                for (size_t q = 0; q < Qsize; q++) {
                                    for (size_t r = 0; r < Rsize; r++, buf += Ssize) {
                                        if constexpr(FillJ) {
                                            const std::complex<double>* __restrict__ D_rs = row_ptr(D_J, r + Rao) + Sao;
                                            std::complex<double>* __restrict__ J_rs = row_ptr(J_RS, r + Rlo) + Slo;
                                            const std::complex<double> D_pq = row_ptr(D_J, p + Pao)[q + Qao] * J_prefactor;
                                            std::complex<double> J_pq{};
                                            for (size_t s = 0; s < Ssize; s++) {
                                                J_pq += D_rs[s] * buf[s];
                                                J_rs[s] += D_pq * buf[s];
                                            }
                                            row_ptr(J_PQ, p + Plo)[q + Qlo] += J_pq * J_prefactor;
                                        }
                                        if constexpr(FillK) {
                                            const QuartetIndex lo{p + Plo, q + Qlo, r + Rlo, Slo};
                                            const QuartetIndex ao{p + Pao, q + Qao, r + Rao, Sao};
                                            exchange.accumulate_row(lo, ao, Ssize, buf, K_prefactor);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (!touched) continue;

            // Stripe task-local J/K blocks into the global matrices. J lives in the
            // (aa) block; each exchange block knows its own spin block offset.
            if constexpr(FillJ) {
                stripe_block(*J, J_PQ, Pfuncs, Qfuncs);
                stripe_block(*J, J_RS, Rfuncs, Sfuncs);
            }

            if constexpr(FillK) {
                exchange.stripe(*K, Pfuncs, Qfuncs, Rfuncs, Sfuncs);
            }
        }
    }

    // => Recover the halves that were never accumulated <= //

    if constexpr(FillJ) {
        symmetrize(*J, nbf);
        if constexpr (SpinBlocked) {
            // J_aa and J_bb are identical: J only depends on the summed charge
            // density D_aa + D_bb.
            for (size_t p = 0; p < nbf; p++) {
                for (size_t q = 0; q < nbf; q++) {
                    (*J)(p + nbf, q + nbf) = (*J)(p, q);
                }
            }
        }
    }

    // K is Hermitian for a Hermitian density. This also fills K_ba = K_ab^H.
    if constexpr(FillK && Hermitian) hermitize(*K, K->dim(0));

    num_computed_shells_ += computed_shells;
    timer_off("build_JK_matrices");
}

void ComplexDirectJK::postiterations() { /*no-op*/ }

void ComplexDirectJK::print_header() const {
    std::string screen_type = options_.get_str("SCREENING");
    if (print_) {
        outfile->Printf("  ==> ComplexDirectJK: Integral-Direct J/K Matrices <==\n\n");

        outfile->Printf("    J tasked:          %11s\n", (do_J_ ? "Yes" : "No"));
        outfile->Printf("    K tasked:          %11s\n", (do_K_ ? "Yes" : "No"));
        outfile->Printf("    Memory [MiB]:      %11ld\n", (memory_ *8L) / (1024L * 1024L));
        outfile->Printf("    Screening Type:    %11s\n", screen_type.c_str());
        outfile->Printf("    Screening Cutoff:  %11.0E\n", cutoff_);
        outfile->Printf("\n");
    }
}

} // namespace psi
