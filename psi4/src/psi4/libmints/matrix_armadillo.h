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

#pragma once

#include "psi4/libmints/matrix.h"

#ifdef USING_OpenOrbitalOptimizer
#ifdef USING_LAPACK_MKL
#include <mkl.h>
#define ARMA_USE_MKL
#define ARMA_USE_MKL_TYPES
#endif
#define ARMA_DONT_USE_FORTRAN_HIDDEN_ARGS
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

namespace psi {
namespace linalg {

/// Copy a Psi4 Matrix irrep block into an Armadillo matrix.
PSI_API arma::mat to_armadillo_matrix(const Matrix& matrix, int h = 0);

/// Copy data from an Armadillo matrix into a Psi4 Matrix irrep block.
PSI_API void from_armadillo_matrix(Matrix& matrix, const arma::mat& m, int h = 0);

}  // namespace linalg
}  // namespace psi
#endif
