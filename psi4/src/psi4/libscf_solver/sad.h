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

#ifndef LIBSCF_SAD_H
#define LIBSCF_SAD_H

#ifdef USING_OpenOrbitalOptimizer
#include "psi4/libmints/matrix.h"
#include <openorbitaloptimizer/scfsolver.hpp>
#endif

//namespace arma {
// class mat;
// class uvec;
//}

namespace psi {

class BasisSet;
class Molecule;
class JK;

namespace scf {

class SADGuess {
   protected:
    int print_;
    int debug_;

    std::shared_ptr<Molecule> molecule_;
    std::shared_ptr<BasisSet> basis_;
    std::vector<std::shared_ptr<BasisSet>> atomic_bases_;
    std::vector<std::shared_ptr<BasisSet>> atomic_fit_bases_;
    SharedMatrix<double> AO2SO_;

    Options& options_;

    SharedMatrix<double> Da_;
    SharedMatrix<double> Db_;
    SharedMatrix<double> Ca_;
    SharedMatrix<double> Cb_;

    std::unique_ptr<JK> jk;

    void common_init();

    void run_atomic_calculations(SharedMatrix<double>& D_AO, SharedMatrix<double>& Huckel_C, SharedVector& Huckel_E);
    void form_gradient(SharedMatrix<double> grad, SharedMatrix<double> F, SharedMatrix<double> D, SharedMatrix<double> S, SharedMatrix<double> X);
    void get_uhf_atomic_density(std::shared_ptr<BasisSet> atomic_basis, std::shared_ptr<BasisSet> fit_basis,
                                SharedVector occ_a, SharedVector occ_b, SharedMatrix<double> D, SharedMatrix<double> Chuckel,
                                SharedVector Ehuckel);
    void get_uhf_atomic_density_ooo(std::shared_ptr<BasisSet> atomic_basis, std::shared_ptr<BasisSet> fit_basis,
                                SharedVector occ_a, SharedVector occ_b, SharedMatrix<double> D, SharedMatrix<double> Chuckel,
                                SharedVector Ehuckel);
    void form_C_and_D(SharedMatrix<double> X, SharedMatrix<double> F, SharedMatrix<double> C, SharedVector E, SharedMatrix<double> Cocc,
                      SharedVector occ, SharedMatrix<double> D);

    void form_D();
    void form_C();

#ifdef USING_OpenOrbitalOptimizer
    std::pair<double, std::vector<arma::mat>> fock_builder(const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm, const std::vector<std::vector<arma::uvec>> & lm_indices, const std::vector<arma::mat> & X, const arma::mat & S, const arma::mat & coreH);
#endif

   public:
    SADGuess(std::shared_ptr<BasisSet> basis, std::vector<std::shared_ptr<BasisSet>> atomic_bases, Options& options);
    virtual ~SADGuess();

    void compute_guess();

    SharedMatrix<double> Da() const { return Da_; }
    SharedMatrix<double> Db() const { return Db_; }
    SharedMatrix<double> Ca() const { return Ca_; }
    SharedMatrix<double> Cb() const { return Cb_; }

    SharedMatrix<double> huckel_guess(bool updated_rule);

    void set_atomic_fit_bases(std::vector<std::shared_ptr<BasisSet>> fit_bases) { atomic_fit_bases_ = fit_bases; }
    void set_print(int print) { print_ = print; }
    void set_debug(int debug) { debug_ = debug; }

};
}  // namespace scf
}  // namespace psi

#endif
