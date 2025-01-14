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

class GHF: public Wavefunction {
    public:
	//GHF(SharedWavefunction ref_wfn, std::shared_ptr<SuperFunctional> functional, Options& options);
	SharedWavefunction ref_wfn;
        
        GHF(SharedWavefunction ref_wfn, std::shared_ptr<SuperFunctional> funct, Options& options,
       std::shared_ptr<PSIO> psio);

        ~GHF() override;

	
	//std::shared_ptr<VBase> V_potential() const override { return potential_; };
 
        //void common_init(SharedWavefunction ref_wfn, Options& options);
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
	auto do_diis();
	void test_jk(Options& options, MintsHelper mints);
	void form_Cocc();
	void form_D();
	auto get_Cocc_block(int i);
        void init_JK(Options& options, MintsHelper mints);
        void build_JK(Options& options, MintsHelper mints);
        double compute_energy(Options& options, MintsHelper mints);
        void clear_tensors();
        void clear_JK_tensors();

    private:
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
	std::shared_ptr<SuperFunctional> functional_;

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

    protected:
        std::shared_ptr<UV> potential_;

	void common_init();

        /// Sizes of each irrep
        std::vector<int> irrep_sizes_;
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

class DIIS {
    public:
        void build_arrays(int max_diis);
	int diis_dim;
        auto do_diis(auto F_, auto D_, auto F1, auto Fp_);
	std::vector<einsums::BlockTensor<std::complex<double>, 2>> F_vecs;
        std::vector<einsums::Tensor<std::complex<double>, 2>> e_vecs;

	std::vector<int> irrep_sizes_;
	size_t nirrep_;
        einsums::BlockTensor<std::complex<double>, 2> X_;
        einsums::BlockTensor<double, 2> S_;

};
}
}
