#pragma once

#include "einsums.hpp"

#include <vector>
#include <functional>
#include "psi4/pybind11.h"


#include "psi4/libfock/v.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/psi4-dec.h"
#include "psi4/libpsio/psio.hpp"

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/sobasis.h"
#include "psi4/libmints/vector.h"
#include "hf.h"

namespace psi {
class Vector;
class JK;
class PCM;
class SuperFunctional;
class BasisSet;
class DIISManager;
class PSIO;
namespace scf {

class GHF : public Wavefunction {
   protected:
        std::vector<int> irrep_sizes_;

        std::shared_ptr<SuperFunctional> functional_;
        SharedWavefunction ref_wfn;

        /// The one electron integrals
        einsums::BlockTensor<std::complex<double>, 2> H_;
        /// The overlap matrix
        einsums::BlockTensor<double, 2> S_;

        /// The inverse square root of the overlap matrix
        einsums::BlockTensor<std::complex<double>, 2> X_;
        einsums::BlockTensor<double, 2> X_real;

        /// The Fock Matrix
        einsums::BlockTensor<std::complex<double>, 2> F_;
        einsums::BlockTensor<std::complex<double>, 2> V_;
        einsums::BlockTensor<std::complex<double>, 2> G_;
        einsums::BlockTensor<std::complex<double>, 2> F1;
        einsums::BlockTensor<std::complex<double>, 2> LEvecs_;
        einsums::BlockTensor<std::complex<double>, 2> REvecs_;
        einsums::BlockTensor<std::complex<double>, 1> Evals_;
        einsums::BlockTensor<double, 1> RealEvals_;
        einsums::BlockTensor<std::complex<double>, 2> F0_;
        einsums::BlockTensor<std::complex<double>, 2> Fp_;
        einsums::BlockTensor<std::complex<double>, 2> J_;
        einsums::BlockTensor<std::complex<double>, 2> K_;

        /// The transformed Fock matrix
        einsums::BlockTensor<std::complex<double>, 2> Fta_, Ftb_;
        /// The MO coefficients
        einsums::BlockTensor<std::complex<double>, 2> C_;
        einsums::BlockTensor<std::complex<double>, 2> C_unsorted;
        /// The occupied MO coefficients
        einsums::BlockTensor<std::complex<double>, 2> Cocc_;
        /// The density matrix
        einsums::BlockTensor<std::complex<double>, 2> D_;
        /// The ubiquitous JK object
        std::shared_ptr<psi::JK> jk_;
        
	SharedMatrix H_mat;
        SharedMatrix V_mat;
        SharedMatrix S_mat;
        SharedMatrix T_mat;
        SharedMatrix F_mat;
        SharedMatrix I_mat;
        SharedMatrix G_mat;
        SharedMatrix X_mat;
        SharedMatrix Sp_mat;
        SharedMatrix Fp_mat;

        int nocc_;
        double nso_;
        double Enuc;

        double E0; // Initial energy
        double E;  // Final energy
        Dimension nsopi_;
        Dimension nalphapi_;
        double nalpha_;
        std::vector<int> occ_per_irrep_;
        std::shared_ptr<psi::BasisSet> basisset_;
        std::shared_ptr<psi::SOBasisSet> sobasisset_;
        size_t nirrep_;
        std::shared_ptr<psi::Molecule> molecule_;
        std::vector<int> occ_per_irrep;
        std::shared_ptr<Matrix> evecs_;
        std::shared_ptr<Vector> evals_;

        std::shared_ptr<Matrix> C_mat;

        SharedMatrix Cocc_mat;

   public:
	std::shared_ptr<GHF> c1_deep_copy(std::shared_ptr<BasisSet> basis);

        void common_init();

        void form_H();
        void form_V();
        void form_S();
        void form_T();
        void form_F();
        void form_X();
        void form_G();
        void form_Sp();
        void form_Fp();
        void setup_potential();
        void compute_potential();
        void diag_eigens();
        void declare_eigens();
        void orthogonalize_fock();
        void diagonalize_fock();
        void make_real_evals();
        void sort_real_evals();
        void evals_sanity_check();
        void back_transform();
        auto get_transpose(auto Tensor, int dimA, int dimB);
        auto MakeComplexBlock(auto A);
        void SCF(Options& options, MintsHelper mints);
        void test_jk(Options& options, MintsHelper mints);
        void form_Cocc();
        void form_D();
        auto get_Cocc_block(int i);
        void init_JK(Options& options, MintsHelper mints);
        void build_JK(Options& options, MintsHelper mints);
        double compute_energy(Options& options, MintsHelper mints);
        void clear_tensors();
        void clear_JK_tensors();

        GHF(SharedWavefunction ref_wfn, std::shared_ptr<SuperFunctional> funct, Options& options,
       std::shared_ptr<PSIO> psio);

        ~GHF() override;
};

}
}


