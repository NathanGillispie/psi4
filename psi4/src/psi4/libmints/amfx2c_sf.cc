/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2025 The Psi4 Developers.
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

#include "psi4/psifiles.h"
#include "psi4/psi4-dec.h"
#include "psi4/physconst.h"

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libfock/points.h"
#include "psi4/libfock/cubature.h"

#include "psi4/libmints/amfx2c_sf.h"
#include <set>

namespace psi {

SFamfX2C::SFamfX2C(std::shared_ptr<Molecule> mol, std::shared_ptr<BasisSet> basis) : mol_(mol), basis_(basis) {}
SFamfX2C::~SFamfX2C() {}

/** TODO **
 * - [ ] Test projecting to AO/SO orbitals
 * - [ ] Add regular tests for grids (e.g. integrating (2π)^{-3/2}\exp(-r^2 / 2) gives 1)
 * - [ ] Create database of radial grids for C++ (mutable on python side?)
 * - [ ] Eventually: build spherical grid *from* (r, f(r)) list? Is that even possible?
 */
void SFamfX2C::compute() {
    //find unique atoms
    std::vector<int> mol_Z(mol_->natom());
    for (int A = 0; A < mol_->natom(); A++) {
        mol_Z[A] = mol_->Z(A);
    }
    std::set<int> unique_Z(mol_Z.begin(), mol_Z.end());

    std::vector<std::shared_ptr<std::vector<MassPoint>>> atomic_grids(0);

    for (const int &Z : unique_Z) {
        // How to get basis for this atom only?

        auto atomic_grid = get_atomic_grid(Z);
        atomic_grids.push_back(atomic_grid);

        int npoints = atomic_grid->size();

        auto x_ = std::make_shared<Vector>(npoints);
        auto y_ = std::make_shared<Vector>(npoints);
        auto z_ = std::make_shared<Vector>(npoints);
        auto w_ = std::make_shared<Vector>(npoints);

        int grid_vector_index = 0;
        for (const auto &Q : *atomic_grid) {
            x_->set(grid_vector_index, Q.x);
            y_->set(grid_vector_index, Q.y);
            z_->set(grid_vector_index, Q.z);
            w_->set(grid_vector_index, Q.w);
            ++grid_vector_index;
        }

        // Get basis functions
        auto extents = std::make_shared<BasisExtents>(basis_, 1.0E-12);
        auto points = std::make_shared<BlockOPoints>(x_, y_, z_, w_, extents);
        const int local_nbf = points->local_nbf();
        BasisFunctions bf{basis_, npoints, local_nbf};
        bf.compute_functions(points);
        SharedMatrix phi = bf.basis_value("PHI");

        auto S = std::make_shared<Matrix>(basis_->nbf(), basis_->nbf());
        S->zero();

        const auto &bf_map = points->functions_local_to_global();
        for (int l_mu = 0; l_mu < local_nbf; l_mu++) {
            int mu = bf_map[l_mu];
            for (int l_nu = 0; l_nu < local_nbf; l_nu++) {
                int nu = bf_map[l_nu];
                for (int p = 0; p < npoints; p++) {
                    S->add(mu, nu, phi->get(p, l_nu) * phi->get(p, l_mu) * w_->get(p));
                }
            }
        }

        S->print();
    }

    // auto spherical_grids_ = get_spherical_grids();
    //
    // auto extents = std::make_shared<BasisExtents>(primary_, epsilon);
    // block(max_points, min_points, max_radius);
    //
    // BasisFunctions pworker(basis_, max_points, max_funcs);
    // pworker.set_deriv(0);
    // pworker.compute_functions(block);
    // auto phi = pworker.basis_value("PHI");
}

}
