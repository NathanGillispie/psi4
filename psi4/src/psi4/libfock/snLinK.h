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

#ifndef SNLINK_H
#define SNLINK_H

#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

#ifdef USING_gauxc
  #include <gauxc/types.hpp>
  #include <gauxc/util/environment.hpp>

  #include <gauxc/xc_integrator.hpp>
  #include <gauxc/xc_integrator/impl.hpp>
  #include <gauxc/xc_integrator/integrator_factory.hpp>

  #include <gauxc/molgrid/defaults.hpp>

  #include <gauxc/molecular_weights.hpp>

  #include <eigen3/Eigen/Core>
#endif

#include "SplitJK.h"

namespace psi {
class Molecule;

/**
 * @brief constructs the K matrix using the GauXC implementation of the
 * seminumerical Linear Exchange (sn-LinK) algorithm,
 * doi: https://doi.org/10.1063/5.0151070
 */
class PSI_API snLinK : public SplitJK {
    // => general Psi4 settings <= //
    // are we doing an incremental Fock build this iteration?
    bool incfock_iter_;

  #ifdef USING_gauxc
    // use Eigen for matrix inputs to GauXC
    // perhaps this can be changed later
    using matrix_type = Eigen::MatrixXd;

    // => Cartesian-Spherical Transformation stuff <= //
    /// Cartesian<->AO basis change matrix used for force-cartesian snLinK transforms. Stored as
    /// CartAO->AO (nbf x nao); applied via transform/back_transform to move densities/K between representations.
    SharedMatrix cartao_to_ao_matrix_;

    // => Gaussian-CCA Transformation stuff <= //
    bool is_cca_;

    std::optional<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> > permutation_matrix_;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> generate_permutation_matrix(
        const std::shared_ptr<BasisSet> psi4_basisset);

    // => Semi-Numerical Stuff <= //
    // are we running snLinK on GPUs?
    bool use_gpu_;
    // what grid pruning scheme is being used?
    std::string pruning_scheme_;
    // what radial quadrature scheme is being used?
    std::string radial_scheme_;
    // how many radial points for the grid?
    size_t radial_points_;
    // how many spherical/angular points for the grid?
    size_t spherical_points_;
    // basis cutoff
    double basis_tol_;

    /// Factory for generating GauXC "Load Balancer" objects
    std::unique_ptr<GauXC::LoadBalancerFactory> gauxc_load_balancer_factory_;

    /// Factory for generating GauXC "Weights Module" objects
    std::unique_ptr<GauXC::MolecularWeightsFactory> gauxc_mol_weights_factory_;

    /// GauXC integrator for actually performing snLinK
    GauXC::IntegratorSettingsSNLinK integrator_settings_;
    std::unique_ptr<GauXC::XCIntegratorFactory<matrix_type> > integrator_factory_;
    std::shared_ptr<GauXC::XCIntegrator<matrix_type> > integrator_;

    // => Psi4 -> GauXC conversion functions <= //
    GauXC::Molecule psi4_to_gauxc_molecule(std::shared_ptr<Molecule> psi4_molecule);
    template<typename T> GauXC::BasisSet<T> psi4_to_gauxc_basisset(std::shared_ptr<BasisSet> psi4_basisset, double basis_tol, bool force_cartesian);

    // => Psi4 -> GauXC enum mappings <= //
    std::tuple<
        std::unordered_map<std::string, GauXC::PruningScheme>,
        std::unordered_map<std::string, GauXC::RadialQuad>
    > generate_enum_mappings();

    // => Other useful stuff <= //
    /// Eigen matrix printout format
    Eigen::IOFormat format_;

    // maximum supported AM for current GauXC instance
    int gauxc_max_am_;
  #endif

  public:
    // => Constructors < = //

    /**
     * @param primary primary basis set for this system.
     *        AO2USO transforms will be built with the molecule
     *        contained in this basis object, so the incoming
     *        C matrices must have the same spatial symmetry
     *        structure as this molecule
     */
    snLinK(std::shared_ptr<BasisSet> primary, Options& options);
    /// Destructor
    ~snLinK() override;

    /// Build the exchange (K) matrix using sn-LinK
    /// primary reference is https://doi.org/10.1063/5.0151070
    void build_G_component(std::vector<std::shared_ptr<Matrix> >& D,
                 std::vector<std::shared_ptr<Matrix> >& G_comp,
         std::vector<std::shared_ptr<TwoBodyAOInt> >& eri_computers) override;

    // => Knobs <= //
    /**
    * Print header information regarding JK
    * type on output file
    */
    void print_header() const override;

    /**
    * print name of method
    */
    std::string name() override { return "sn-LinK"; }

    // setters and getters
    void set_incfock_iter(bool incfock_iter) { incfock_iter_ = incfock_iter; }

    int get_max_am() {
      #ifdef USING_gauxc
        return gauxc_max_am_;
      #else
        throw PSIEXCEPTION("Psi4 is not installed with GauXC support!");
      #endif
    }
};

}

#endif
