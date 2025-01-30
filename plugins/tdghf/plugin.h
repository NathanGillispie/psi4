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
#include "psi4/libscf_solver/hf.h"

namespace psi {
namespace tdghf {

class GHF {
    public:
	//GHF(SharedWavefunction ref_wfn, std::shared_ptr<SuperFunctional> functional, Options& options);
	SharedWavefunction ref_wfn;
	std::shared_ptr<SuperFunctional> functional;
        void common_init(SharedWavefunction ref_wfn, Options& options);
	void form_H();
	void form_V();
	void form_S();
	void form_T();
        void form_F();
	void form_X();
	void form_G();
        void form_Sp();
	void form_Fp();
	void init_td();
        void diag_eigens();
	void declare_eigens();
	void orthogonalize_fock();
	void diagonalize_fock();
	void diagonalize_A();
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
        void build_JK(Options& options, MintsHelper mints);
	void init_JK(Options& options, MintsHelper mints);
        //auto build_JK(Options& options, MintsHelper mints, auto JKwK_, auto D_);
        double compute_energy(Options& options, MintsHelper mints, auto JKwK_);
        void clear_tensors();
        void clear_JK_tensors();
	SharedMatrix G_mat;
        Dimension nsopi_;
	Dimension nbetapi_;
	Dimension nelecpi_;
	//std::vector<int> nelec_;
	int nelec;
        std::shared_ptr<psi::BasisSet> basisset_;
        size_t nirrep_;
        std::vector<int> irrep_sizes_;
        std::vector<std::shared_ptr<TwoBodyAOInt>> ints;

    private:
	SharedMatrix H_mat;
	SharedMatrix V_mat;
	SharedMatrix S_mat;
	SharedMatrix T_mat;
	SharedMatrix F_mat;
	SharedMatrix I_mat;
        //SharedMatrix G_mat;
	SharedMatrix X_mat;
        SharedMatrix Sp_mat;
        SharedMatrix Fp_mat;
        
        int nocc_;	
	double nso_;
	double Enuc;
	double E0; // Initial energy
	double E;  // Final energy
	Dimension nalphapi_;
	double nbeta_;
	double nalpha_;
	std::vector<int> occ_per_irrep_;
	//std::shared_ptr<psi::BasisSet> basisset_;
	std::shared_ptr<psi::SOBasisSet> sobasisset_;
	std::shared_ptr<psi::Molecule> molecule_;
        std::vector<int> occ_per_irrep;
	std::shared_ptr<Matrix> evecs_;
	std::shared_ptr<Vector> evals_;

	std::shared_ptr<Matrix> C_mat;

	SharedMatrix Cocc_mat;

    protected:
        SharedMatrix Gaa_, Gab_, Gba_, Gbb_, Jar_, Jai_, Kai_, Kar_, Kb_, wKa_, wKb_;
        SharedMatrix Jaar_, Jaai_,  Jabr_, Jabi_,  Jbar_, Jbai_,  Jbbr_, Jbbi_, J_, K_;
        SharedMatrix Kaar_, Kaai_,  Kabr_, Kabi_,  Kbar_, Kbai_,  Kbbr_, Kbbi_;

        /// Sizes of each irrep
	/// The one electron integrals
        einsums::BlockTensor<std::complex<double>, 2> H_;
        /// The overlap matrix
        einsums::BlockTensor<double, 2> S_;

        /// The inverse square root of the overlap matrix
        einsums::BlockTensor<std::complex<double>, 2> X_;
        einsums::BlockTensor<double, 2> X_real;

        /// The Fock Matrix
        einsums::BlockTensor<std::complex<double>, 2> F_;
        einsums::BlockTensor<std::complex<double>, 2> EINT_;

	einsums::BlockTensor<std::complex<double>, 4> G_;
        einsums::BlockTensor<std::complex<double>, 2> F1;
	einsums::BlockTensor<std::complex<double>, 2> LEvecs_;
        einsums::BlockTensor<std::complex<double>, 2> REvecs_;
        einsums::BlockTensor<std::complex<double>, 1> Evals_;
	einsums::BlockTensor<double, 1> RealEvals_;
	einsums::BlockTensor<std::complex<double>, 2> F0_;
        einsums::BlockTensor<std::complex<double>, 2> Fp_;


        einsums::BlockTensor<std::complex<double>, 2> to_complex_block;


        // TD-GHF ABBA matrix
	einsums::BlockTensor<std::complex<double>, 2> M_;

	// TD-GHF A matrix
	// TODO do this on the fly in M_ rather than separate MA_ matrix. Same with MB_
	einsums::BlockTensor<std::complex<double>, 2> MA_;
        einsums::BlockTensor<std::complex<double>, 2> MB_;

	//TD-GHF eigenvectors/eigenvalues and other matrices to solve the TDA
	einsums::BlockTensor<std::complex<double>, 2> A1;
        
	einsums::Tensor<std::complex<double>, 2> JK_;

        //Initialize blocks
	einsums::Tensor<std::complex<double>, 2> D_block;
        einsums::Tensor<std::complex<double>, 2> subblock;
        einsums::Tensor<std::complex<double>, 2> Cocc;


	einsums::BlockTensor<std::complex<double>, 2> JKwK_;
        //einsums::BlockTensor<std::complex<double>, 2> K_;

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
        std::shared_ptr<JK> jk_;

        //std::shared_ptr<psi::JK> jk_;
        /// The functional.
        std::shared_ptr<psi::SuperFunctional> func_;
        /// The functional exchange integrator.
        std::shared_ptr<psi::VBase> v_;

};
class GJK: public GHF {
    public:
        GJK() = default;

        // Declare G_mat as SharedMatrix
        SharedMatrix G_mat;
        einsums::BlockTensor<std::complex<double>, 2> G_;

        // The correct signature for the function
        void initialize_gjk(auto mints);
        //auto build_JK(auto D_, auto nsopi_, auto nirrep_, auto irrep_sizes_);
        //auto build_JK(auto D_);
};
}
}
