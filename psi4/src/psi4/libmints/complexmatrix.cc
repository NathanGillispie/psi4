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

// For outfile in ComplexMatrix::print()
#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include <ostream>

namespace psi {

#ifdef USING_Einsums

/// Overload for Dimension callers
ComplexMatrix::ComplexMatrix(const std::string& name, const Dimension& row_sizes, const Dimension& col_sizes)
  : tensor_(name, row_sizes.blocks(), col_sizes.blocks()) {}


const Dimension ComplexMatrix::rowspi() const {
    return Dimension(tensor_.tile_size(0));
}

const Dimension ComplexMatrix::colspi() const {
    return Dimension(tensor_.tile_size(1));
}

std::shared_ptr<ComplexMatrix> ComplexMatrix::clone() const {
    return std::make_shared<ComplexMatrix>(*this);
}

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

// self -= other
void ComplexMatrix::subtract(const ComplexMatrix& other) {
    axpy(-1.0, other);
}

// Re(Tr(self^H other)), summed over diagonal tiles
double ComplexMatrix::vector_dot(const ComplexMatrix& other) const {
    const auto& other_t = static_cast<const TiledT&>(other);
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

// Raw per-tile complex sub-blocks to a PSIO file, mirroring Matrix::save with
// SaveType::SubBlocks (libmints/matrix.cc).
void ComplexMatrix::save(std::shared_ptr<PSIO>& psio, size_t fileno) {
    bool already_open = psio->open_check(fileno);
    if (!already_open) psio->open(fileno, PSIO_OPEN_OLD);

    for (int h = 0; h < static_cast<int>(tensor_.grid_size(0)); ++h) {
        if (!tensor_.has_tile(h, h) || tensor_.has_zero_size(h, h)) continue;
        auto& t = tensor_.tile(h, h);
        std::string entry = tensor_.name() + " Tile " + std::to_string(h);
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
        if (tensor_.tile_size(0)[h] == 0 || tensor_.tile_size(1)[h] == 0) continue;
        auto& t = tensor_.tile(h, h);  // lazily allocated to the declared size
        std::string entry = tensor_.name() + " Tile " + std::to_string(h);
        psio->read_entry(fileno, entry, (char*)t.data(), sizeof(std::complex<double>) * t.size());
    }

    if (!already_open) psio->close(fileno, 1);
}

void ComplexMatrix::print(std::string out, const char *extra) const {
    if (extra != nullptr) { throw PSIEXCEPTION("Not implemented"); }

    std::shared_ptr<psi::PsiOutStream> printer = (out == "outfile" ? outfile : std::make_shared<PsiOutStream>(out));

    einsums::fprintln(*printer->stream(), tensor_);
}

#endif  // USING_Einsums

}  // namespace psi
