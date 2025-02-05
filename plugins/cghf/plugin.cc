/*
 * @BEGIN LICENSE
 *
 * cghf by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2023 The Psi4 Developers.
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
#include <malloc.h>
#include <stdio.h>

#include "einsums.hpp"
#include <map>
#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <complex>
#include <utility>
#include <bits/stdc++.h>
#include "psi4/libfock/jk.h"
#include "psi4/libfock/v.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/factory.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/sobasis.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"
#include "psi4/psi4-dec.h"
#include "einsums/LinearAlgebra.hpp"
#include "einsums/_Common.hpp"
#include "einsums/_Index.hpp"
#include <cmath>
#include <functional>
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/matrix.h"
#include "psi4/lib3index/dftensor.h"
#include "psi4/libmints/sointegral_onebody.h"
#include "psi4/libmints/sointegral_twobody.h"

#include "plugin.h"

#pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in) \
    initializer(omp_priv = std::complex<double>(0.0, 0.0))

using namespace einsums;
using namespace einsums::tensor_algebra;


namespace psi{
namespace cghf {

void GJK::initialize_gjk(auto mints) {
    G_mat = mints.ao_eri();
}


extern "C" PSI_API
int read_options(std::string name, Options& options)
{
    if (name == "TDGHF"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
    }

    return true;
}


extern "C" PSI_API
SharedWavefunction cghf(SharedWavefunction ref_wfn,
		          //const std::shared_ptr<SuperFunctional> &functional,
	        	  Options& options) {
    GHF cghf;
    int print = options.get_int("PRINT");
        /* Your code goes here */
    //basisset_ = ref_wfn->basisset();
    //std::shared_ptr<psi::SOBasisSet> sobasisset_ = ref_wfn->sobasisset();
    //int nirrep_ = ref_wfn->nirrep();
    //std::shared_ptr<psi::Molecule> molecule_ = ref_wfn->molecule();

    //MintsHelper mints(MintsHelper(ref_wfn->basisset(), options, 0));
    //auto func_ = functional;
    cghf.common_init(ref_wfn, options);
    //cghf.form_V();

    /*
    // Get initial integrals
    auto integral = std::make_shared<IntegralFactory>(basisset_, basisset_, basisset_, basisset_);

    // Nuclear repulsion energy
    auto Enuc = molecule_->nuclear_repulsion_energy({0,0,0});
    std::cout << Enuc << "\n";

    // Initialize mints
    auto mints = std::make_shared<MintsHelper>(basisset_);
    */
    
    // Typically you would build a new wavefunction and populate it with data
    return ref_wfn;
}


void GHF::common_init(SharedWavefunction ref_wfn, Options& options){
    timer_on("Psi4 integrals");
    MintsHelper mints(MintsHelper(ref_wfn->basisset(), options, 0));
    V_mat  = mints.so_potential();
    S_mat  = mints.so_overlap();
    H_mat  = mints.so_kinetic();
    T_mat  = mints.so_kinetic();
    //T_mat->print("outfile");
    //V_mat->print("outfile");
    F_mat  = T_mat->clone();
    F_mat->add(V_mat);
    Dipole_mat = mints.so_dipole()[2];
    Angular_mat = mints.so_angular_momentum()[2];

    //F_mat->print();
    X_mat  = S_mat->clone();
    Sp_mat = S_mat->clone();
    Fp_mat = F_mat->clone();
    //G_mat = mints.ao_eri();
    timer_off("Psi4 integrals");
    timer_on("Common init Psi4");
    //G_mat->print();
    nso_        = ref_wfn->nso();
    nsopi_      = ref_wfn->nsopi();
    nalpha_     = ref_wfn->nalpha();
    nalphapi_   = ref_wfn->nalphapi();
    basisset_   = ref_wfn->basisset();
    sobasisset_ = ref_wfn->sobasisset();
    nirrep_     = ref_wfn->nirrep();
    molecule_   = ref_wfn->molecule();
    nbeta_      = ref_wfn->nbeta();
    nbetapi_    = ref_wfn->nbetapi();
    nelecpi_    = nalphapi_ + nbetapi_;
    auto factory_    = mints.factory();

    //molecule_->print();
    Enuc = molecule_->nuclear_repulsion_energy({0,0,0});
    Jaai_ = SharedMatrix(factory_->create_matrix("K alpha"));
    Jabi_ = SharedMatrix(factory_->create_matrix("K beta"));
    Jbai_ = SharedMatrix(factory_->create_matrix("K beta"));
    Jbbi_ = SharedMatrix(factory_->create_matrix("K beta"));

    Kaai_ = SharedMatrix(factory_->create_matrix("K beta"));
    Kabi_ = SharedMatrix(factory_->create_matrix("K beta"));
    Kbai_ = SharedMatrix(factory_->create_matrix("K beta"));
    Kbbi_ = SharedMatrix(factory_->create_matrix("K beta"));

    Jaar_ = SharedMatrix(factory_->create_matrix("K alpha"));
    Jabr_ = SharedMatrix(factory_->create_matrix("K beta"));
    Jbar_ = SharedMatrix(factory_->create_matrix("K beta"));
    Jbbr_ = SharedMatrix(factory_->create_matrix("K beta"));

    Kaar_ = SharedMatrix(factory_->create_matrix("K beta"));
    Kabr_ = SharedMatrix(factory_->create_matrix("K beta"));
    Kbar_ = SharedMatrix(factory_->create_matrix("K beta"));
    Kbbr_ = SharedMatrix(factory_->create_matrix("K beta"));

    jk_ = JK::build_JK(ref_wfn->basisset(), std::shared_ptr<BasisSet>(), options);

    jk_->initialize();

    /*
    int charge = molecule_->molecular_charge();
    nelec = 0;
    for (int i = 0; i < molecule_->natom(); ++i) {
      nelec += (int)molecule_->Z(i);
    }
    
    nelec -= charge;
    int nbocc_ = (nelec - molecule_->multiplicity() + 1) / 2;
    int naocc_ = nbocc_ + molecule_->multiplicity() - 1;
    //Assume RHF for now

    int nocc = nelec/2;
    int nalpha = nocc;
    */

    timer_off("Common init Psi4");
    timer_on("Common init Einsums");
    form_S();
    REvecs_ = einsums::BlockTensor<std::complex<double>, 2>("R Eigenvectors", irrep_sizes_);
    LEvecs_ = einsums::BlockTensor<std::complex<double>, 2>("L Eigenvectors", irrep_sizes_);
    JKwK_ = einsums::BlockTensor<std::complex<double>, 2>("JKwK_", irrep_sizes_);

    Evals_ = einsums::BlockTensor<std::complex<double>, 1>("Eigenvalues", irrep_sizes_);
    RealEvals_ = einsums::BlockTensor<double, 1>("Real Eigenvalues", irrep_sizes_);
    //G_ = einsums::BlockTensor<std::complex<double>, 2>("DUMB", irrep_sizes_);
    MA_ = einsums::BlockTensor<std::complex<double>, 2>("TD A matrix", irrep_sizes_);
    MB_ = einsums::BlockTensor<std::complex<double>, 2>("TD B matrix", irrep_sizes_);

    F_ = einsums::BlockTensor<std::complex<double>, 2>("AO Fock", irrep_sizes_);
    EINT_ = einsums::BlockTensor<std::complex<double>, 2>("Kinetic Matrix", irrep_sizes_);

    Fp_ = einsums::BlockTensor<std::complex<double>, 2>("AO Fock Ortho", irrep_sizes_);

    D_ = einsums::BlockTensor<std::complex<double>, 2>("AO Density", irrep_sizes_);
    C_ = einsums::BlockTensor<std::complex<double>, 2>("C", irrep_sizes_);
    Cocc_ = einsums::BlockTensor<std::complex<double>, 2>("Cocc", irrep_sizes_);
    //J_ = einsums::BlockTensor<std::complex<double>, 2>("J", irrep_sizes_);
    //K_ = einsums::BlockTensor<std::complex<double>, 2>("K", irrep_sizes_);

    C_.zero();
    F1 = einsums::BlockTensor<std::complex<double>, 2>("Temp Fock Matrix", irrep_sizes_);
    F1.zero();
    REvecs_.zero();
    LEvecs_.zero();
    Evals_.zero();
    RealEvals_.zero();
    F_.zero();
    Fp_.zero();
    D_.zero();
    C_.zero();
    Cocc_.zero();
    JKwK_.zero();
    //J_.zero();
    //K_.zero();
    G_.zero();
   // Empty orthogonalized Fock Matrix
    F0_ = einsums::BlockTensor<std::complex<double>, 2>("Orthogonalized Fock Matrix", irrep_sizes_);
    F0_.zero();
    
    /*
    TD_REvecs_ = einsums::BlockTensor<std::complex<double>, 2>("TD R Eigenvectors", irrep_sizes_);
    TD_LEvecs_ = einsums::BlockTensor<std::complex<double>, 2>("TD L Eigenvectors", irrep_sizes_);

    TD_Evals_ = einsums::BlockTensor<std::complex<double>, 1>("TD Eigenvalues", irrep_sizes_);
    TD_RealEvals_ = einsums::BlockTensor<double, 1>("TD Real Eigenvalues", irrep_sizes_);
    */

    MA_.zero();
    MB_.zero();
    A1.zero();
    timer_off("Common init Einsums");
    //TD_REvecs_.zero();
    //TD_LEvecs_.zero();
    //TD_Evals_.zero();
    SCF(options, mints);
    //compute_kinetic_E();
    //init_td();
    //diagonalize_A();
}



/*
void GHF::form_G() {
   timer_on("Forming G");
   G_.zero();
   for (int i = 0; i < nirrep_; i++) {
       int row_dim = G_mat->rowdim(i);
       int col_dim = G_mat->coldim(i);
       auto G_block_super = einsums::Tensor<std::complex<double>, 2>(molecule_->irrep_labels().at(i), row_dim, col_dim);
       G_block_super.zero();

       for (int j = 0; j < row_dim; j++) {
           for (int k = 0; k < col_dim; k++) {
	       G_block_super(j, k) = G_mat->get(i, j, k);
	   }
       G_.push_block(G_block_super);
       }
   }
   timer_off("Forming G");
}
*/

void GHF::form_S() {
    S_mat->set_name("Overlap Matrix");

    for (int i = 0; i < S_mat->nirrep(); i++) {
        irrep_sizes_.push_back(2*S_mat->coldim(i));
        auto S_block = einsums::Tensor<double, 2>(
                       "S block", 2*S_mat->rowdim(i), 2*S_mat->coldim(i));

        S_block.zero();
        for (int j = 0; j < S_mat->rowdim(i); j++) {
                for (int k = 0; k < S_mat->coldim(i); k++) {
                        S_block(j, k) = S_mat->get(i, j, k);
                        S_block(j+S_mat->rowdim(i), k+S_mat->coldim(i)) = S_mat->get(i, j, k);
                }
        }
	S_.push_block(S_block);

    }
}


void GHF::form_X() {
    //std::cout << "Forming X\n";
    //auto X_mat = einsums::linear_algebra::pow(S_, -0.5);
    //X_mat->print();
    //X_mat->power(-0.5, 1e-14);
    //X_mat->print();
    auto X_real = einsums::linear_algebra::pow(S_, -0.5, 1e-14);

    for (int i = 0; i < nirrep_; i++) {
        auto X_block = einsums::Tensor<std::complex<double>, 2>("X_block", irrep_sizes_[i], irrep_sizes_[i]);
	X_block.zero();
        for (int j = 0; j < irrep_sizes_[i]; j++) {
            for (int k = 0; k < irrep_sizes_[i]; k++) {
                 X_block(j, k) = X_real[i](j, k);
            }
        }
        X_.push_block(X_block);
    }
}

void GHF::form_F() {
    //std::cout << "Forming F\n";
    //F_mat->add(V_mat);
    for (int i = 0; i < nirrep_; i++) {
        auto F_block = einsums::Tensor<std::complex<double>, 2>(
                      molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]);


        F_block.zero();
        for (int j = 0; j < F_mat->rowdim(i); j++) {
            for (int k = 0; k < F_mat->coldim(i); k++) {
                    F_block(j,k) = {F_mat->get(i, j, k), -0.5*Angular_mat->get(i, j, k)*1e-3};
                    F_block(j+F_mat->rowdim(i), k+F_mat->coldim(i)) = {F_mat->get(i, j, k), -0.5*Angular_mat->get(i, j, k)*1e-3};
            }
        }
          F0_[i] = F_block;
    }
}

void GHF::orthogonalize_fock() {
   F1.zero();
   einsums::linear_algebra::gemm<false, false>(std::complex<double>{1.0}, F_, X_, std::complex<double>{0.0},  &F1);

   einsums::linear_algebra::gemm<true, false>(std::complex<double>{1.0}, X_, F1, std::complex<double>{0.0}, &Fp_);
   
}

auto GetRealVector(auto A, auto dim) {
    auto real = einsums::Tensor<double, 1>("R", dim);

    for (int i = 0; i < dim; i++) {
        real(i) = A(i).real();
    }

    return real;

}

void GHF::diagonalize_fock() {
   timer_on("Diagonalizing Fock matrix");
   auto Fp_blocks = Fp_.vector_data();
   for (int i = 0; i < nirrep_; i++) {
       auto Fp_irrep = Fp_blocks[i];
       if (irrep_sizes_[i] > 0) {
           einsums::linear_algebra::geev(&Fp_irrep, &Evals_[i], &LEvecs_[i], &REvecs_[i]);
       }


    }
    auto eval_blocks = Evals_.vector_data();

    for (int i = 0; i < nirrep_; i++) {
        auto eval_irrep = eval_blocks[i];
        auto real_eval_irrep = GetRealVector(eval_irrep, irrep_sizes_[i]);
        RealEvals_[i] = real_eval_irrep;
    }
   timer_off("Diagonalizing Fock matrix");

}

void GHF::back_transform() {
   einsums::linear_algebra::gemm<false, false>(std::complex<double>{1.0}, X_, REvecs_, std::complex<double>{0.0},  &C_);
}

// Comparator to sort pairs based on the second element (value)
bool comparePairs(const std::pair<int, double>& a, const std::pair<int, double>& b) {
    return a.second < b.second;
}

auto bubbleSort(std::map<int, double>& map) {
    // Convert map to vector of pairs for sorting
    std::vector<std::pair<int, double>> vec(map.begin(), map.end());

    int n = vec.size();
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (vec[j].second > vec[j + 1].second) {
                std::swap(vec[j], vec[j + 1]);
                swapped = true;
            }
        }
        if (!swapped)
            break;
    }

   return vec;
}


void GHF::sort_real_evals(){
    // Sorted Evecs
    auto Eval_blocks = RealEvals_.vector_data();
    C_unsorted = C_;

    for (int i = 0; i < nirrep_; i++) {
            std::map<int, double> eval_map_temp;
            int orderpi [irrep_sizes_[i]];
            for (int j = 0; j < 2*nsopi_[i]; j++) {
                double eval = Eval_blocks[i](j);
                eval_map_temp.insert(std::pair(j, eval));
            }
            auto eval_map = bubbleSort(eval_map_temp);
            
	    int counter = 0; 
	    for (const auto& pair : eval_map) {
	        orderpi[counter] = pair.first;
		counter += 1;
	    }
	    for (int j = 0; j < irrep_sizes_[i]; j++) { 
	        for (int k = 0; k < irrep_sizes_[i]; k++) {
		    C_[i](j, k) = C_unsorted[i](j, orderpi[k]);
		    RealEvals_[i](k) = Eval_blocks[i](orderpi[k]);
		}	
	        }
	    }
    
}

void GHF::form_D() {
    timer_on("Matts density matrix");
    auto C_block = C_.vector_data();
    D_.zero();
    Cocc_.zero();
    for (int i = 0; i < nirrep_; i++) {
        auto Cocc = einsums::Tensor<std::complex<double>, 2>("Cocc", irrep_sizes_[i], nelecpi_[i]);
        auto cCocc = einsums::Tensor<std::complex<double>, 2>("Cocc", irrep_sizes_[i], nelecpi_[i]);

	Cocc.zero();
	cCocc.zero();
        for (int j = 0; j < irrep_sizes_[i]; j++) {
                for (int k = 0; k < nelecpi_[i]; k++) {
                        Cocc(j,k) = C_block[i](j, k);
			cCocc(j, k) = {C_block[i](j, k).real(), -C_block[i](j, k).imag()};
                        //std::cout << Cocc(j,k) << "\n";

                }
        }
        auto D_block = einsums::Tensor<std::complex<double>, 2>("Temp D", irrep_sizes_[i], irrep_sizes_[i]);
	D_block.zero();
        einsums::linear_algebra::gemm<false, true>(std::complex<double>{1.0, 0.0}, Cocc, cCocc, std::complex<double>{0.0},  &D_block);
        D_[i] = D_block;
	Cocc_[i] = Cocc;
    }
    timer_off("Matts density matrix");
}


auto GetConjugate(auto A) {
    int dim = A.dim(0);
    auto to_conj = einsums::Tensor<std::complex<double>, 2>("Conjugate", dim, dim);

    to_conj.zero();

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
	    to_conj(i, j) = {A(i, j).real(), -A(i,j).imag()};
	
	}
    }


    return to_conj;
}

auto MakeComplex(auto A) { 
    auto dim = A.dim(0);
    auto to_complex = einsums::Tensor<std::complex<double>, 2>("C", dim, dim);
    to_complex.zero();

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
	    to_complex(i, j) = A(i, j);
	
	}
    }

    return to_complex;
  
}

auto GHF::MakeComplexBlock(auto A) {
    auto to_complex_block = einsums::BlockTensor<std::complex<double>, 2>("CB", irrep_sizes_);
    to_complex_block.zero();

    for (int i = 0; i < nirrep_; i++) {
	auto subblock = einsums::Tensor<std::complex<double>, 2>("CBB", irrep_sizes_[i]);
	subblock.zero();

        for (int j = 0; j < irrep_sizes_[i]; j++) {
	    for (int k = 0; k < irrep_sizes_[i]; k++) {
	        subblock(j, k) = A(j, k);
	    }
	}

	to_complex_block[i] = subblock;
    }

    return to_complex_block;

}

auto GetReal(auto A, auto dimA, auto dimB) {
    auto real = einsums::Tensor<double, 2>("R", dimA, dimB);
    for (int i = 0; i < dimA; i++) {
        for (int j = 0; j < dimB; j++) {
            real(i,j) = A(i,j).real();
        }
    }

    return real;

}

auto GetImag(auto A, auto dimA, auto dimB) {
    auto imag = einsums::Tensor<double, 2>("R", dimA, dimB);
    for (int i = 0; i < dimA; i++) {
        for (int j = 0; j < dimB; j++) {
            imag(i,j) = A(i,j).imag();
        }
    }

    return imag;

}



//auto GJK::build_JK(auto D_, auto nsopi_, auto nirrep_, auto irrep_sizes_) {
void GHF::build_JK(Options& options, MintsHelper mints) {
    //auto JKwK_ = einsums::BlockTensor<std::complex<double>, 2>("JK matrix", irrep_sizes_);
    //JKwK_->zero();

    //auto J_ = einsums::BlockTensor<std::complex<double>, 2>("J matrix", irrep_sizes_);
    //auto K_ = einsums::BlockTensor<std::complex<double>, 2>("K matrix", irrep_sizes_);
    double cutoff_ = 1.0E-12;
    //auto JK_ = einsums::BlockTensor<std::complex<double>, 2>("JK matrix", irrep_sizes_);
    //J_.zero();
    //K_.zero();
    //JK_.zero();
    //println(J_);
    timer_on("Matts JK");
    JKwK_.zero();
    // Push the C matrix on
    std::vector<SharedMatrix>& Cl = jk_->C_left();
    std::vector<SharedMatrix>& Cr = jk_->C_right();

    //Diagonal C matrices
    timer_on("Forming C subsets");
    auto Cai_subset = std::make_shared<Matrix>(nirrep_, nsopi_, nelecpi_);
    auto Car_subset = std::make_shared<Matrix>(nirrep_, nsopi_, nelecpi_);
    auto Cbi_subset = std::make_shared<Matrix>(nirrep_, nsopi_, nelecpi_);
    auto Cbr_subset = std::make_shared<Matrix>(nirrep_, nsopi_, nelecpi_);

    Cai_subset->zero();
    Car_subset->zero();
    Cbi_subset->zero();
    Cbr_subset->zero();

    Jaar_->zero();
    Jaai_->zero();
    Jbbr_->zero();
    Jbbi_->zero();

    Kaar_->zero();
    Kaai_->zero();
    Kabr_->zero();
    Kabi_->zero();
    Kbar_->zero();
    Kbai_->zero();
    Kbbr_->zero();
    Kbbi_->zero();

    Cl.clear();
    Cr.clear(); 

    for (int i = 0; i < nirrep_; i++) {
        for (int j = 0; j < nsopi_[i]; j++) {
            for (int k = 0; k < nelecpi_[i]; k++) {
                Car_subset->set(i, j, k, Cocc_[i](j, k).real());
                Cai_subset->set(i, j, k, Cocc_[i](j, k).imag());
                Cbr_subset->set(i, j, k, Cocc_[i](j+nsopi_[i], k).real());
                Cbi_subset->set(i, j, k, Cocc_[i](j+nsopi_[i], k).imag());

	        /*	
		std::complex<double> Cjk = Cocc_[i](j, k);
                std::complex<double> Ckj = Cocc_[i](k, j);
                Car_subset->set(i, j, k, Cjk.real());
                Cai_subset->set(i, j, k, Cjk.imag());
                Cbr_subset->set(i, j, k, Cjk.real());
                Cbi_subset->set(i, j, k, Cjk.imag());
		*/
                }
            }
        }
    timer_off("Forming C subsets");

    /*
     Ca Ca
     Ca Ca^
     Ca^ Ca
     Ca^ Ca^
     *
    */
    timer_on("Jaa/Kaa");
    Cl.push_back(Car_subset);
    Cl.push_back(Car_subset);
    Cl.push_back(Cai_subset);
    Cl.push_back(Cai_subset);
    
    timer_on("Cr push back"); 
    Cr.push_back(Car_subset);
    Cr.push_back(Cai_subset);
    Cr.push_back(Car_subset);
    Cr.push_back(Cai_subset);
    timer_off("Cr push back");

    jk_->set_do_wK(false);    
    jk_->compute();
    auto J = jk_->J();
    auto K = jk_->K();

    Jaar_->copy(J[0]);
    Jaar_->add(J[3]);

    Jaai_->copy(J[2]);
    Jaai_->subtract(J[1]);

    Kaar_->copy(K[0]);
    Kaar_->add(K[3]);

    Kaai_->copy(K[2]);
    Kaai_->subtract(K[1]);
    
    //Cl.clear();
    
    Cr.clear();
    timer_off("Jaa/Kaa");
    /*
      Ca Cb
      Ca Cb^
      Ca^ Cb
      Ca^ Cb^
    */

    timer_on("Kab");
    Cr.push_back(Cbr_subset);
    Cr.push_back(Cbi_subset);
    Cr.push_back(Cbr_subset);
    Cr.push_back(Cbi_subset);

    jk_->set_do_wK(false);
    jk_->compute();
    J = jk_->J();
    K = jk_->K();

    //Jabr_->copy(J[0]);
    //Jabr_->add(J[3]);

    //Jabi_->copy(J[1]);
    //Jabi_->add(J[2]);

    Kabr_->copy(K[0]);
    Kabr_->add(K[3]);

    Kabi_->copy(K[2]);
    Kabi_->subtract(K[1]);

    Cl.clear();
    Cr.clear();

    timer_off("Kab");
    /*
      Cb Ca
      Cb Ca^
      Cb^ Ca
      Cb^ Ca^
    */

    /*
    timer_on("Kba");
    Cl.push_back(Cbr_subset);
    Cl.push_back(Cbr_subset);
    Cl.push_back(Cbi_subset);
    Cl.push_back(Cbi_subset);

    Cr.push_back(Car_subset);
    Cr.push_back(Cai_subset);
    Cr.push_back(Car_subset);
    Cr.push_back(Cai_subset);

    jk_->set_do_wK(false);
    jk_->set_do_J(false);
    jk_->compute();
    K = jk_->K();

    //Jbar_->copy(J[0]);
    //Jbar_->add(J[3]);

    //Jbai_->copy(J[1]);
    //Jbai_->add(J[2]);

    Kbar_->copy(K[0]);
    Kbar_->add(K[3]);

    Kbai_->copy(K[2]);
    Kbai_->subtract(K[1]);

    Cl.clear();
    Cr.clear();
    
    timer_off("Kba");
      Cb Cb
      Cb Cb^
      Cb^ Cb
      Cb^ Cb^
    */
    
    /*
    timer_on("Kbb");
    Cl.push_back(Cbr_subset);
    Cl.push_back(Cbr_subset);
    Cl.push_back(Cbi_subset);
    Cl.push_back(Cbi_subset);

    Cr.push_back(Cbr_subset);
    Cr.push_back(Cbi_subset);
    Cr.push_back(Cbr_subset);
    Cr.push_back(Cbi_subset);
   
    jk_->set_do_wK(false); 
    jk_->set_do_J(false);
    jk_->compute();
    //J = jk_->J();
    K = jk_->K();
    
    Jbbr_->copy(J[0]);
    Jbbr_->add(J[3]);

    Jbbi_->copy(J[2]);
    Jbbi_->subtract(J[1]);
    Kbbr_->copy(K[0]);
    Kbbr_->add(K[3]);

    Kbbi_->copy(K[2]);
    Kbbi_->subtract(K[1]);
    
    timer_off("Kbb");
    */

    for (int i = 0; i < nirrep_; i++) {
        for (int p = 0; p < nsopi_[i]; p++) {
            for (int q = 0; q < nsopi_[i]; q++) {
                //Jaa
		JKwK_[i](p, q) += std::complex<double>(Jaar_->get(i, p, q), Jaai_->get(i, p, q));
		JKwK_[i](p, q) += std::complex<double>(Jaar_->get(i, p, q), Jaai_->get(i, p, q));

		//Jbb
		JKwK_[i](p+nsopi_[i], q+nsopi_[i]) += std::complex<double>(Jaar_->get(i, p, q), Jaai_->get(i, p, q));
                JKwK_[i](p+nsopi_[i], q+nsopi_[i]) += std::complex<double>(Jaar_->get(i, p, q), Jaai_->get(i, p, q));

		//Kaa
		JKwK_[i](p, q) -= std::complex<double>(Kaar_->get(i, p, q), Kaai_->get(i, p, q));

		//Kbb
		JKwK_[i](p+nsopi_[i], q+nsopi_[i]) -= std::complex<double>(Kaar_->get(i, p, q), Kaai_->get(i, p, q));

		//Kab
		JKwK_[i](p, q+nsopi_[i]) -= std::complex<double>(Kabr_->get(i, p, q), Kabi_->get(i, p, q));

		//Kba
		JKwK_[i](p+nsopi_[i], q) -= std::complex<double>(Kabr_->get(i, q, p), -Kabi_->get(i, q, p));
            }
        }
    }
    //auto jkmat = jk_->D();
    //jkmat->print_out();
    
    /* 
    for (int i = 0; i < nirrep_; i++) {
        auto *data = D_[i].data();
	for (int p = 0; p < nsopi_[i]; p++) {
            for (int q = 0; q < nsopi_[i]; q++) {
                for (int r = 0; r < nsopi_[i]; r++) {
                    for (int s = 0; s < nsopi_[i]; s++) {
                        int pq = p*nsopi_[i] + q;
                        int rs = r*nsopi_[i] + s;
                        int ps = p*nsopi_[i] + s;
                        int rq = r*nsopi_[i] + q;
                        double pqrs = {G_mat->get(i, pq, rs)};
                        double psrq = {G_mat->get(i, ps, rq)};
			//std::complex<double>* pqrs_ptr = &pqrs;
			//std::complex<double>* psrq_ptr = &psrq;
                         
			std::complex<double>* D_aa = &D_[i](s, r);
                        std::complex<double>* D_ab = &D_[i](s, r+nsopi_[i]);
			std::complex<double>* D_ba = &D_[i](s+nsopi_[i], r);
			std::complex<double>* D_bb = &D_[i](s+nsopi_[i], r+nsopi_[i]);
			if (std::abs(pqrs) > cutoff_) {
		            //Jaa gaaaa Daa
                            //auto D_aa  = D_[i](s, r);
                            auto J_contract_aa = pqrs * *D_aa;
                            JKwK_[i](p, q) += J_contract_aa;
			
			    //Jaa gaabb Dbb
                            //auto D_bb = D_[i](s+nsopi_[i], r+nsopi_[i]);
                            auto J_contract_bb = pqrs * *D_bb;
                            JKwK_[i](p, q) += J_contract_bb;

                            //Jbb gbbbb Dbb
                            JKwK_[i](p+nsopi_[i], q+nsopi_[i]) += J_contract_bb;

                            //Jbb gbbaa Daa
                            JKwK_[i](p+nsopi_[i], q+nsopi_[i]) += J_contract_aa;
			}
			if (std::abs(psrq) > cutoff_) {
			    //Kaa gaaa Da
                            auto K_contract_aa = psrq * *D_aa;
                            JKwK_[i](p, q) -= K_contract_aa;

                            //Kab gaabb Dab
                            //auto D_ab = D_[i](s, r+nsopi_[i]);
                            auto K_contract_ab = psrq * *D_ab;
                            JKwK_[i](p, q+nsopi_[i]) -= K_contract_ab;

                            //Kba gbbaa Dba
                            //auto D_ba = D_[i](s+nsopi_[i], r);
                            auto K_contract_ba = psrq * *D_ba;
                            JKwK_[i](p+nsopi_[i], q) -= K_contract_ba;

                            //Kbb gbbbb Dbb
                            auto K_contract_bb = psrq * *D_bb;
                            JKwK_[i](p+nsopi_[i], q+nsopi_[i]) -= K_contract_bb;
			}

                    }
                }
            }
        }
    }
    */
    timer_off("Matts JK");
}
 
	   
void GHF::evals_sanity_check() {
     double running = -10000000000.0;

     for (int i = 0; i < nirrep_; i++) {
         for (int j = 0; j < irrep_sizes_[i]; j++) {
	     double reval = RealEvals_[i](j);

	     if (reval >= running) {
	        running = reval;
	     }
	     else {
		 std::cout << "Eigenvalues are not sorted at irrep " << i << ", index " << j << " (value=" << reval << ")\n"; 
	     }
	 }
     
     }
}

double GHF::compute_energy(Options& options, MintsHelper mints, auto JKwK_) {
    auto TensorE = einsums::Tensor<std::complex<double>, 0>("E");
    auto temp = einsums::BlockTensor<std::complex<double>, 2>("temp", irrep_sizes_);
    //auto JKwK_ = (*JKwK_);

    temp = F0_;
    for (int i = 0; i < nirrep_; i++) {
        auto JK_block = JKwK_[i];
	//auto K_block = K_[i];
	
        JK_block *= 0.5;

        temp[i] += JK_block;
    }
   
    einsums::tensor_algebra::einsum(
        0.0, einsums::tensor_algebra::Indices{}, &TensorE, 1.0,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        D_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        temp);


    std::complex<double> E_complex = (std::complex<double>)TensorE;
    E = E_complex.real();
    //std::cout << "Energy: " << std::setprecision(11) << E;

    return E;
}

void GHF::SCF(Options& options, MintsHelper mints) {
    timer_on("SCF");
    bool diis_selected = false;
    form_X();
    form_F();
    //form_G();
    int maxiter = 20;
    double scf_thresh = 1e-10;
    int max_diis = 6; //diis_max
    int total_diis = 0; //diis_count
    int e_rms = 1e6;
    double E = 0;
    //auto gjk = GJK();
    
    //gjk.initialize_gjk(mints);
    //JKwK_ = einsums::BlockTensor<std::complex<double>, 2>("JKwK_", irrep_sizes_);
    //JKwK_.zero();

    for (int i = 0; i < maxiter; i++) {
        F_.zero();
        F_ += F0_;
	F_ += JKwK_;
	//delete JKwK_;
        //F_ += J_;
        //F_ -= K_;
        orthogonalize_fock();
	diagonalize_fock();
        back_transform();
        sort_real_evals();
 
        form_D();
        build_JK(options, mints);
	//auto JK_ = gjk.build_JK(D_, nsopi_, nirrep_, irrep_sizes_);
	//JKwK_ = JK_;
        double newE = compute_energy(options, mints, JKwK_);
	double diffE = std::abs(E-newE);
	//std::cout << E << " " << newE << "\n";
	std::cout << "Iter " << i+1 << " | " << std::setprecision(15) << E+Enuc << " | deltaE = " << diffE << "\n";
	E = newE;

	if (diffE < scf_thresh) {
		std::cout << "\nConverged!\n";

		break;
	    }
	}
    timer_off("SCF");
    //println(RealEvals_);

}

void GHF::init_td() {
   /*
   std::cout << "TD initialized\n";
   double spin = nalpha_ - nbeta_;
   double mult = 2.0*spin + 1.0;
   for (int n = 0; n < nirrep_; n++) {
       int row_dim = G_mat->rowdim(n);
       int col_dim = G_mat->coldim(n);


       // Only do TDA for now
       //auto M_block = einsums::Tensor<std::complex<double>, 2>(molecule_->irrep_labels().at(n), row_dim, col_dim);
       
       //
       // A B
       // B A
       // 
       // Oke dokey no big dealio 
       // There appeared to be a big dealio with the indices. 48x48 here but 14x14 for J_ and K_
       // Let's just do the TDA for now
       //
       auto MA_block = einsums::Tensor<std::complex<double>, 2>(molecule_->irrep_labels().at(n), row_dim, col_dim);

       for (int i = 0; i < nsopi_[n]; i++) {
	   for (int j = 0; j < nsopi_[n]; j++) {
              //nbf - nelectrons
	      for (int a = 0; a < nsopi_[n]; a++) {
	         for (int b = 0; b < nsopi_[n]; b++) { 
	           std::complex<double> Aiajb = (0.0, 0.0);
                   int pq = i * nsopi_[n] + j;    // Mapping for the J_ matrix
                   int rs = a * nsopi_[n] + b;    // Mapping for the K_ matrix
                   int ps = i * nsopi_[n] + b;    // Mapping for the second term
                   int rq = a * nsopi_[n] + j;    // Mapping for the second term
		   
		   
		   std::complex<double> pqrs = J_[n](pq, rs);
                   std::complex<double> psrq = K_[n](ps, rq);

		   // (ia||jb)
		   auto iajb = (pqrs - psrq);
		   Aiajb += iajb;
		   // Kronecker Deltas tell us the term only matters when
		   // i = j and a = b
	           if (i == j & a == b) {
		       Aiajb += (RealEvals_(a) - RealEvals_(i));
		   }
		   std::cout << pq << " " << rs << " " << pq+nsopi_[n] << " " << rs+nsopi_[n] << "\n"; 
		   MA_block(pq, rs) = Aiajb;
                   MA_block(pq + nsopi_[n], rs + nsopi_[n]) = Aiajb;
                   //MA_block(pq, rs + nsopi_[n]) = Aiajb;
                   //MA_block(pq + nsopi_[n], rs) = Aiajb;
		   // Both A matrices
                   //M_block(pq, rs) = Aiajb;          
		   //M_block(pq+nsopi_[n], rs+nsopi_[n]) = Aiajb;
		   
                   // Both B matrices
		   //M_block(pq, rs+nsopi_[n]) = iajb;
		   //M_block(pq+nsopi_[n], rs) = iajb;

                   //std::cout << pq << " " << rs << " " << ps << " " << rq << "\n";
                   //MA_block(pq, rs) = Aiajb;
		   //MB_block(pq, rs) = (pqrs - prqs);
		 }
              }
           }
	   MA_[n] = MA_block;
	   //println(MA_block);
	   //MA_.push_block(MA_block);
       }

   }
   */
}

void GHF::diagonalize_A() {
   //Come up with Davidson diagonalizer to diagonalize, no geev
   auto A_blocks = MA_.vector_data();
   for (int i = 0; i < nirrep_; i++) {
       /*
       auto TD_REvecs_ = einsums::Tensor<std::complex<double>, 2>("TD R Eigenvectors", 2*irrep_sizes_[i]);
       auto TD_LEvecs_ = einsums::Tensor<std::complex<double>, 2>("TD L Eigenvectors", 2*irrep_sizes_[i]);

       auto TD_Evals_ = einsums::Tensor<std::complex<double>, 1>("TD Eigenvalues", irrep_sizes_);
       auto TD_RealEvals_ = einsums::Tensor<double, 1>("TD Real Eigenvalues", irrep_sizes_);

       auto A_irrep = MA_[i];
       if (irrep_sizes_[i] > 0) {
           einsums::linear_algebra::geev(&A_irrep, &TD_Evals_[i], &TD_LEvecs_[i], &TD_REvecs_[i]);
	   }
       */


    }
    //auto eval_blocks = TD_Evals_.vector_data();
    /*
    for (int i = 0; i < nirrep_; i++) {
        auto eval_irrep = eval_blocks[i];
        auto real_eval_irrep = GetRealVector(eval_irrep, irrep_sizes_[i]);
        TD_RealEvals_[i] = real_eval_irrep;
    }

    println(TD_RealEvals_);
    */
}



}} // End namespaces










