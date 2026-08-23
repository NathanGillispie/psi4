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

#include "psi4/libmints/matrix_armadillo.h"

#ifdef USING_OpenOrbitalOptimizer

namespace psi {
namespace linalg {

arma::mat to_armadillo_matrix(const Matrix& matrix, int h) {
    int nc = matrix.coldim(h);
    int nr = matrix.rowdim(h);
    arma::mat m(nr, nc);
    for (int ir = 0; ir < nr; ir++)
        for (int ic = 0; ic < nc; ic++)
            m(ir, ic) = matrix.get(h, ir, ic);
    return m;
}

void from_armadillo_matrix(Matrix& matrix, const arma::mat& m, int h) {
    int nc = matrix.coldim(h);
    int nr = matrix.rowdim(h);
    for (int ir = 0; ir < nr; ir++)
        for (int ic = 0; ic < nc; ic++)
            matrix.set(h, ir, ic, m(ir, ic));
}

}  // namespace linalg
}  // namespace psi

#endif
