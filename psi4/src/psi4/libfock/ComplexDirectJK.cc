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

#include "jk.h"
#include "psi4/libmints/basisset.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"

using namespace psi;

namespace psi {

ComplexDirectJK::ComplexDirectJK(std::shared_ptr<BasisSet> primary, Options& options) : JK(primary), options_(options) { common_init(); }

ComplexDirectJK::~ComplexDirectJK() {}

void ComplexDirectJK::common_init() {}

size_t ComplexDirectJK::memory_estimate() {
    return 0;  // Placeholder
}

void ComplexDirectJK::print_header() const {
    if (print_) {
        outfile->Printf("  ==> ComplexDirectJK: Integral-Direct J/K Matrices (Einsums, Placeholder) <==\n\n");
        outfile->Printf("    J tasked:          %11s\n", (do_J_ ? "Yes" : "No"));
        outfile->Printf("    K tasked:          %11s\n", (do_K_ ? "Yes" : "No"));
        outfile->Printf("    wK tasked:         %11s\n", (do_wK_ ? "Yes" : "No"));
        outfile->Printf("\n");
    }
}

void ComplexDirectJK::preiterations() { }

void ComplexDirectJK::postiterations() { }

void ComplexDirectJK::compute_JK() { }

}  // namespace psi
