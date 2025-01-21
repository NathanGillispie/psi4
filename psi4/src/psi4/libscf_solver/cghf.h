#pragma once

#include "einsums.hpp"

#include "psi4/libfock/v.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/psi4-dec.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/sobasis.h"
#include "psi4/libmints/vector.h"
#include "hf.h"

namespace psi {
namespace scf {

class CGHF: public HF {
    public:
	//GHF(SharedWavefunction ref_wfn, std::shared_ptr<SuperFunctional> functional, Options& options);
	SharedWavefunction ref_wfn;
        CGHF(SharedWavefunction ref_wfn, std::shared_ptr<SuperFunctional> functional);
        CGHF(SharedWavefunction ref_wfn, std::shared_ptr<SuperFunctional> functional, Options& options,
            std::shared_ptr<PSIO> psio);
        ~CGHF() override;
        //std::shared_ptr<VBase> V_potential() const override { return potential_; };

    	double get_energies() { return E; }

        std::shared_ptr<VBase> V_potential() const override { return potential_; };


        //void common_init(SharedWavefunction ref_wfn, Options& options);
	void form_V() override;
	void finalize() override;
	void form_S();
	void form_T();
	void form_H() override;
        void form_F() override;
	void form_init_F();
	void form_X();
	void form_G() override;
	void form_2e();
        void form_Sp();
	void form_Fp();
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

        //Placeholders
	void save_density_and_energy();

	auto MakeComplexBlock(auto A);
	void SCF(Options& options, MintsHelper mints);
	void test_jk(Options& options, MintsHelper mints);
	void form_C(double shift=0.0) override;
	void form_D() override;
	auto get_Cocc_block(int i);
        void init_JK(Options& options, MintsHelper mints);
        void build_JK(Options& options, MintsHelper mints);
        double compute_E();
        //double compute_energy(Options& options, MintsHelper mints);
	void clear_tensors();
        void clear_JK_tensors();
        void form_initial_C() { form_C(); }

        double compute_kinetic_E();
	double compute_1e_E();
	double compute_coulomb_E();

    private:
	SharedMatrix H_mat;
	SharedMatrix V_;
	//SharedMatrix S_;
	SharedMatrix T_;
	SharedMatrix F_mat;
	SharedMatrix I_mat;
        SharedMatrix G_mat;
	SharedMatrix X_mat;
        SharedMatrix Sp_mat;
        SharedMatrix Fp_mat;
        
	double nso_;
	double Enuc;
	double E0; // Initial energy
	double E;  // Final energy
	//Dimension nsopi_;
	int nalphapi_;
	double nalpha_;
	std::vector<int> occ_per_irrep_;
	std::shared_ptr<psi::BasisSet> basisset_;
	std::shared_ptr<psi::SOBasisSet> sobasisset_;
	//size_t nirrep_;
	std::shared_ptr<psi::Molecule> molecule_;
        std::vector<int> occ_per_irrep;
	std::shared_ptr<Matrix> evecs_;
	std::shared_ptr<Vector> evals_;
        std::vector<double> nocc_;
	std::shared_ptr<Matrix> C_mat;

	SharedMatrix Cocc_mat;

    protected:
        std::shared_ptr<UV> potential_;
        //Options& options;
	void common_init();
        void setup_potential() override;
        int iteration_;
        bool converged_;
        double nuclearrep_;

        /// Sizes of each irrep
        std::vector<int> irrep_sizes_;
	/// The one electron integrals
        //einsums::BlockTensor<std::complex<double>, 2> EINH_;
        /// The overlap matrix
        einsums::BlockTensor<double, 2> EINS_;

        /// The inverse square root of the overlap matrix
        einsums::BlockTensor<std::complex<double>, 2> X_;
        einsums::BlockTensor<double, 2> X_real;

        /// The Fock Matrix
        einsums::BlockTensor<std::complex<double>, 2> F_;
        einsums::BlockTensor<std::complex<double>, 2> EINT_;
	einsums::BlockTensor<std::complex<double>, 2> twoe_;
        einsums::BlockTensor<std::complex<double>, 2> F1;
	einsums::BlockTensor<std::complex<double>, 2> LEvecs_;
        einsums::BlockTensor<std::complex<double>, 2> REvecs_;
        einsums::BlockTensor<std::complex<double>, 1> Evals_;
        einsums::BlockTensor<double, 1> RealEvals_;
	einsums::BlockTensor<std::complex<double>, 2> F0_;
        einsums::BlockTensor<std::complex<double>, 2> Fp_;
        einsums::Tensor<std::complex<double>, 2> Jaa_;
        einsums::Tensor<std::complex<double>, 2> Jbb_;
        einsums::Tensor<std::complex<double>, 2> Kaa_;
        einsums::Tensor<std::complex<double>, 2> Kab_;
        einsums::Tensor<std::complex<double>, 2> Kba_;
        einsums::Tensor<std::complex<double>, 2> Kbb_;

	/*
        einsums::BlockTensor<std::complex<double>, 2> J1;
        einsums::BlockTensor<std::complex<double>, 2> K1;
        einsums::BlockTensor<std::complex<double>, 2> J2;
        einsums::BlockTensor<std::complex<double>, 2> K2;
        einsums::BlockTensor<std::complex<double>, 2> J3;
        einsums::BlockTensor<std::complex<double>, 2> K3;
        einsums::BlockTensor<std::complex<double>, 2> J4;
        einsums::BlockTensor<std::complex<double>, 2> K4;
	*/
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
        /// The functional.
        std::shared_ptr<psi::SuperFunctional> func_;
        /// The functional exchange integrator.
        std::shared_ptr<psi::VBase> v_;

};

}
}
