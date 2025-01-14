/*
 * @BEGIN LICENSE
 *
 * GHF by Matthew Ward, a plugin to:
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

//#include "einsums/TensorAlgebra.hpp"
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

#include "ghf.h"

using namespace einsums;
using namespace einsums::tensor_algebra;

namespace psi{
namespace scf {

GHF::GHF(SharedWavefunction ref_wfn, std::shared_ptr<SuperFunctional> func, Options& options, std::shared_ptr<PSIO> psio)
    : Wavefunction(options), functional_(func) {
    shallow_copy(ref_wfn);
    psio_ = psio;
    common_init();
}

GHF::~GHF() {}


void GHF::common_init(){
    MintsHelper mints(MintsHelper(ref_wfn->basisset(), options(), 0));
    
    V_mat  = mints.so_potential();
    S_mat  = mints.so_overlap();
    H_mat  = mints.so_kinetic();
    T_mat  = mints.so_kinetic();
    F_mat  = T_mat->clone();
    F_mat->add(V_mat);
    F_mat->print();
    X_mat  = S_mat->clone();
    Sp_mat = S_mat->clone();
    Fp_mat = F_mat->clone();
    G_mat = mints.ao_eri();

    G_mat->print();
    nso_        = ref_wfn->nso();
    nsopi_      = ref_wfn->nsopi();
    nalpha_     = ref_wfn->nalpha();
    nalphapi_   = ref_wfn->nalphapi();
    basisset_   = ref_wfn->basisset();
    sobasisset_ = ref_wfn->sobasisset();
    nirrep_     = ref_wfn->nirrep();
    molecule_   = ref_wfn->molecule();

    molecule_->print();
    Enuc = molecule_->nuclear_repulsion_energy({0,0,0});

    int charge = molecule_->molecular_charge();
    int nelec = 0;

    for (int i = 0; i < molecule_->natom(); ++i) {
	nelec += (int)molecule_->Z(i);
    }

    nelec -= charge;

    //Assume RHF for now

    int nocc = nelec/2;
    int nalpha = nocc;
    form_S();
    REvecs_ = einsums::BlockTensor<std::complex<double>, 2>("R Eigenvectors", irrep_sizes_);
    LEvecs_ = einsums::BlockTensor<std::complex<double>, 2>("L Eigenvectors", irrep_sizes_);

    Evals_ = einsums::BlockTensor<std::complex<double>, 1>("Eigenvalues", irrep_sizes_);
    RealEvals_ = einsums::BlockTensor<double, 1>("Real Eigenvalues", irrep_sizes_);
    G_ = einsums::BlockTensor<std::complex<double>, 2>("DUMB", irrep_sizes_);

    F_ = einsums::BlockTensor<std::complex<double>, 2>("AO Fock", irrep_sizes_);
    Fp_ = einsums::BlockTensor<std::complex<double>, 2>("AO Fock Ortho", irrep_sizes_);

    D_ = einsums::BlockTensor<std::complex<double>, 2>("AO Density", irrep_sizes_);
    C_ = einsums::BlockTensor<std::complex<double>, 2>("C", irrep_sizes_);
    Cocc_ = einsums::BlockTensor<std::complex<double>, 2>("Cocc", irrep_sizes_);
    J_ = einsums::BlockTensor<std::complex<double>, 2>("J", irrep_sizes_);
    K_ = einsums::BlockTensor<std::complex<double>, 2>("K", irrep_sizes_);

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
    J_.zero();
    K_.zero();
    G_.zero();
   // Empty orthogonalized Fock Matrix
    F0_ = einsums::BlockTensor<std::complex<double>, 2>("Orthogonalized Fock Matrix", irrep_sizes_);
    F0_.zero();
    SCF(options(), mints);
}

void GHF::form_G() {
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
}

void GHF::form_S() {
    S_mat->set_name("Overlap Matrix");

    for (int i = 0; i < S_mat->nirrep(); i++) {
        irrep_sizes_.push_back(2*S_mat->coldim(i));
        auto S_block = einsums::Tensor<double, 2>(
                        molecule_->irrep_labels().at(i), 2*S_mat->rowdim(i), 2*S_mat->coldim(i));

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
        if (irrep_sizes_[i] == 0) {
           continue;
        }
        for (int j = 0; j < irrep_sizes_[i]; j++) {
            for (int k = 0; k < irrep_sizes_[i]; k++) {
                 X_block(j, k) = X_real[i](j, k);
            }
        }
        X_.push_block(X_block);
    }
}

void GHF::form_H() {
    //Placeholder for now
}

void GHF::form_F() {
    //std::cout << "Forming F\n";
    //F_mat->add(V_mat);

    for (int i = 0; i < F_mat->nirrep(); i++) {
        auto F_block = einsums::Tensor<std::complex<double>, 2>(
                      molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]);

        F_block.zero();
        for (int j = 0; j < F_mat->rowdim(i); j++) {
            for (int k = 0; k < F_mat->coldim(i); k++) {
                    F_block(j,k) = F_mat->get(i, j, k);
                    F_block(j+F_mat->rowdim(i), k+F_mat->coldim(i)) = F_mat->get(i, j, k);
            }
        }
          F0_[i] = F_block;
    }
}

void GHF::form_V() {
    //std::cout << "Forming V\n";
    
    for (int i = 0; i < V_mat->nirrep(); i++) {
        auto V_block = einsums::Tensor<std::complex<double>, 2>(
                      molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]);

        V_block.zero();
        for (int j = 0; j < V_mat->rowdim(i); j++) {
            for (int k = 0; k < V_mat->coldim(i); k++) {
                    V_block(j,k) = V_mat->get(i, j, k);
                    V_block(j+V_mat->rowdim(i), k+V_mat->coldim(i)) = V_mat->get(i, j, k);
            }
        }
          V_[i] = V_block;
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
    auto C_block = C_.vector_data();
    D_.zero();
    Cocc_.zero();
    for (int i = 0; i < nirrep_; i++) {
        auto Cocc = einsums::Tensor<std::complex<double>, 2>("Cocc", irrep_sizes_[i], 2*nalphapi_[i]);
	Cocc.zero();
        for (int j = 0; j < irrep_sizes_[i]; j++) {
                for (int k = 0; k < 2*nalphapi_[i]; k++) {
                        Cocc(j,k) = C_block[i](j, k);
                        //std::cout << Cocc(j,k) << "\n";

                }
        }
        auto D_block = einsums::Tensor<std::complex<double>, 2>("Temp D", irrep_sizes_[i], irrep_sizes_[i]);
	D_block.zero();
        einsums::linear_algebra::gemm<false, true>(std::complex<double>{1.0, 0.0}, Cocc, Cocc, std::complex<double>{0.0},  &D_block);
        D_[i] = D_block;
	Cocc_[i] = Cocc;
    }

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



void GHF::build_JK(Options& options, MintsHelper mints) {
    for (int i = 0; i < nirrep_; i++) {
        auto Jaa = einsums::Tensor<std::complex<double>, 2>("Jaa", nsopi_[i], nsopi_[i]);

	auto Jbb = einsums::Tensor<std::complex<double>, 2>("Jbb", nsopi_[i], nsopi_[i]);
        auto Kaa = einsums::Tensor<std::complex<double>, 2>("Kaa", nsopi_[i], nsopi_[i]);
        auto Kab = einsums::Tensor<std::complex<double>, 2>("Kab", nsopi_[i], nsopi_[i]);
        auto Kba = einsums::Tensor<std::complex<double>, 2>("Kba", nsopi_[i], nsopi_[i]);
        auto Kbb = einsums::Tensor<std::complex<double>, 2>("Kbb", nsopi_[i], nsopi_[i]);

        auto J_block = einsums::Tensor<std::complex<double>, 2>("J block", 2*nsopi_[i], 2*nsopi_[i]);
        auto K_block = einsums::Tensor<std::complex<double>, 2>("K block", 2*nsopi_[i], 2*nsopi_[i]);

	Jaa.zero();
	Jbb.zero();
	Kaa.zero();
	Kab.zero();
	Kba.zero();
	Kbb.zero();
	J_block.zero();
	K_block.zero();

        //Jaa and Jbb
	for (int p = 0; p < nsopi_[i]; p++) {
	    for (int q = 0; q < nsopi_[i]; q++) {
		for (int r = 0; r < nsopi_[i]; r++) {
		    for (int s = 0; s < nsopi_[i]; s++) {
			int pq = p*nsopi_[i] + q;
		        int rs = r*nsopi_[i] + s;
			int ps = p*nsopi_[i] + s;
			int rq = r*nsopi_[i] + q;
			
			std::complex<double> pqrs = {G_mat->get(i, pq, rs)};
		        std::complex<double> psrq = {G_mat->get(i, ps, rq)};

			//Jaa gaaaa Daa
			auto D_aa  = D_[i](s, r);
                        auto J_contract_aa = pqrs*D_aa;               
			Jaa(p, q) += J_contract_aa;

			//Jaa gaabb Dbb
			auto D_bb = D_[i](s+nsopi_[i], r+nsopi_[i]);
			auto J_contract_bb = pqrs*D_bb;
			Jaa(p, q) += J_contract_bb;

			//Jbb gbbbb Dbb
			Jbb(p, q) += J_contract_bb;

			//Jbb gbbaa Daa
			Jbb(p, q) += J_contract_aa;

                        //Kaa gaaaa Daa
                        auto K_contract_aa = psrq*D_aa;
			Kaa(p, q) += K_contract_aa;
		        
		        //Kab gaabb Dab
			auto D_ab = D_[i](s, r+nsopi_[i]);	
			auto K_contract_ab = psrq*D_ab;
			Kab(p, q) += K_contract_ab;

			//Kba gbbaa Dba
			auto D_ba = D_[i](s+nsopi_[i], r);
			auto K_contract_ba = psrq*D_ba;
		        Kba(p, q) += K_contract_ba;

		        //Kbb gbbbb Dbb
                        auto K_contract_bb = psrq*D_bb;
			Kbb(p, q) += K_contract_bb;
		    }
		}
	    }
	}
        
	//Fill blocks
        for (int j = 0; j < nsopi_[i]; j++) {
	    for (int k = 0; k < nsopi_[i]; k++) {
	        J_block(j, k) = Jaa(j, k);
		J_block(j+nsopi_[i], k+nsopi_[i]) = Jbb(j, k);
		K_block(j, k) = Kaa(j, k);
		K_block(j, k+nsopi_[i]) = Kab(j, k);
		K_block(j+nsopi_[i], k) = Kba(j, k);
		K_block(j+nsopi_[i], k+nsopi_[i]) = Kbb(j, k);
	    }
	}

	J_[i] = J_block;
	K_[i] = K_block;
    }
}


// Psi4 expects the potential_ variable to be from the VBase class
// This took a whole lot of trial and error (am dumb), but I managed to wrap the
// ScalarPotential class into VBase
// Turns out imposter syndrome gave me an idea
//
/* 
class ScalarPotential : public VBase {
public:
    ScalarPotential(double potential) : potential_(potential) {}

    // Override any necessary methods from VBase
    double get() const { return potential_; }

private:
    double potential_;
};
*/


void GHF::setup_potential() {
    if (functional_->needs_xc()) {
        potential_ = std::make_shared<UV>(functional_, basisset_, options());
        potential_->initialize();
    } else {
        potential_ = nullptr;
    }
}


//TODO find some way to make the shared_ptr complex. Only accepts real values now. No bueno
void GHF::compute_potential() {
    // First take dot product of density matrix and potential matrix
    
    auto TensorV = einsums::Tensor<std::complex<double>, 0>("V");

    auto temp = einsums::BlockTensor<std::complex<double>, 2>("temp", irrep_sizes_);
 
    //QUESTION would we be able to just take the dot product directly
    //or do I need to assign V_ to temp like with Energy?
    /*
    einsums::tensor_algebra::einsum(
        0.0, einsums::tensor_algebra::Indices{}, &TensorV, 1.0,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        D_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        temp);


    std::complex<double> calculated_potential = i(0.0, 0.0); 

    //TODO get trace
    */


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

double GHF::compute_energy(Options& options, MintsHelper mints) {
    auto TensorE = einsums::Tensor<std::complex<double>, 0>("E");

    auto temp = einsums::BlockTensor<std::complex<double>, 2>("temp", irrep_sizes_);
    temp = F0_;
    for (int i = 0; i < nirrep_; i++) {
        auto J_block = J_[i];
	auto K_block = K_[i];
        auto temp_J_block = J_block;
	auto temp_K_block = K_block;
	
	temp_J_block *= 0.5;
	temp_K_block *= 0.5;

        temp[i] += temp_J_block;
	temp[i] -= temp_K_block;
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

void DIIS::build_arrays(int max_diis) {
        F_vecs.resize(max_diis);
        e_vecs.resize(max_diis);
	diis_dim = 0;
}

auto DIIS::do_diis(auto F_, auto D_, auto F1, auto Fp_) {
    //
    //e_i = daggerX[FDS-SDF]X
    //middle = FDS-SDF
    //
    //
    int max_diis = 6; //diis_max
    int total_diis = 0; //diis_count
    //int diis_dim = 0;
    int e_rms = 1e6;
    int curdim = 0;
    double diis_alpha = 0.75;

    //auto e_vec  = einsums::Tensor<double, 3>("Error Vector", 1, 1, 1);
    //auto F_diis = einsums::BlockTensor<std::complex<double>, 2>("F DIIS", irrep_sizes_);
    auto FDS = einsums::BlockTensor<std::complex<double>, 2>("F*D*S",
                                   irrep_sizes_);
    auto SDF = einsums::BlockTensor<std::complex<double>, 2>("S*D*F",
                                   irrep_sizes_);
    auto commutator = einsums::BlockTensor<std::complex<double>, 2>("Commutator",
                                   irrep_sizes_);
    auto ortho_comm_temp = einsums::BlockTensor<std::complex<double>, 2>("Orthogonalized Commutator Temp",
                                   irrep_sizes_);
    auto ortho_comm = einsums::BlockTensor<std::complex<double>, 2>("Orthogonalized Commutator",
                                   irrep_sizes_);
    
    //e_vec.zero();
    //F_diis.zero();
    SDF.zero();
    commutator.zero();
    ortho_comm_temp.zero();
    ortho_comm.zero();
    double emax_array[nirrep_];
    
    for (int irrep = 0; irrep < nirrep_; irrep++) {
        auto FDS_block = einsums::Tensor<std::complex<double>, 2>("F*D*S_sub",
                                   irrep_sizes_[irrep], irrep_sizes_[irrep]);
        auto FDS_block_temp = einsums::Tensor<std::complex<double>, 2>("F*D*S Temp",
                                   irrep_sizes_[irrep], irrep_sizes_[irrep]);
        auto SDF_block = einsums::Tensor<std::complex<double>, 2>("S*D*F_sub",
                                   irrep_sizes_[irrep], irrep_sizes_[irrep]);
        auto SDF_block_temp = einsums::Tensor<std::complex<double>, 2>("S*D*F Temp",
                                   irrep_sizes_[irrep], irrep_sizes_[irrep]);
        auto ortho_comm_block = einsums::Tensor<std::complex<double>, 2>("Ortho Commutator_sub",
                                   irrep_sizes_[irrep], irrep_sizes_[irrep]);
        auto ortho_comm_temp = einsums::Tensor<std::complex<double>, 2>("Ortho Commutator_sub",
                                  irrep_sizes_[irrep], irrep_sizes_[irrep]);
        auto F_block_temp = einsums::Tensor<std::complex<double>, 2>("F DIIS Temp",
                                   irrep_sizes_[irrep], irrep_sizes_[irrep]);
        auto F_block = einsums::Tensor<std::complex<double>, 2>("F DIIS Block",
                                   irrep_sizes_[irrep], irrep_sizes_[irrep]);

	//FDS
        einsums::linear_algebra::gemm<false, false>(std::complex<double>{1.0}, D_[irrep], MakeComplex(S_[irrep]), std::complex<double>{0.0},  &FDS_block_temp);
        einsums::linear_algebra::gemm<false, false>(std::complex<double>{1.0}, F_[irrep], FDS_block_temp, std::complex<double>{0.0}, &FDS_block);
    
        FDS[irrep] = FDS_block;
        
	//SDF
        einsums::linear_algebra::gemm<false, false>(std::complex<double>{1.0}, D_[irrep], F_[irrep], std::complex<double>{0.0},  &SDF_block_temp);
        einsums::linear_algebra::gemm<false, false>(std::complex<double>{1.0}, MakeComplex(S_[irrep]), SDF_block_temp, std::complex<double>{0.0}, &SDF_block);
    
        SDF[irrep] = SDF_block;
        auto conj_SDF = GetConjugate(SDF_block);
        
	//FDS - SDF
        auto commutator = FDS_block;
        commutator -= conj_SDF;
        //Commutator*X
        einsums::linear_algebra::gemm<false, false>(std::complex<double>{1.0}, commutator, X_[irrep], std::complex<double>{0.0},  &ortho_comm_temp);
        
        //daggerX*Commutator
        einsums::linear_algebra::gemm<true, false>(std::complex<double>{1.0}, GetConjugate(X_[irrep]), ortho_comm_temp, std::complex<double>{0.0}, &ortho_comm_block);
        //Get emax by looping
         
	for (int j = 0; j < irrep_sizes_[irrep]; j++) {
           for (int k = 0; k < irrep_sizes_[irrep]; k++) {
               if (std::abs(ortho_comm_block(j, k).real()) > emax_array[irrep]) {
    	           emax_array[irrep] = std::abs(ortho_comm_block(j, k).real());
               }
          }
        } 
	std::cout << "emax: " << emax_array[irrep] << "\n";	
        auto e = ortho_comm_block; 
	if (emax_array[irrep] > 1.2) { 
          auto is_diis = "DAMP"; //Honestly have no idea what this is
          auto i = curdim;
    
          auto F_diis_temp = einsums::Tensor<std::complex<double>, 2>("Temp F DIIS", irrep_sizes_[i], irrep_sizes_[i]);
          F1.zero();
	  Fp_.zero();

	  einsums::linear_algebra::gemm<false, false>(std::complex<double>{1.0}, F_, X_, std::complex<double>{0.0},  &F1);

          einsums::linear_algebra::gemm<true, false>(std::complex<double>{1.0}, X_, F1, std::complex<double>{0.0}, &Fp_);

          F_vecs[i] = Fp_;
    
          if (i == 0) {
             curdim += 1;
	     auto newF = F_vecs[i];
	  }
          else {
             curdim = 0;
             auto temp1 = F_vecs[0];
	     temp1 *= diis_alpha;
	     auto temp2 = F_vecs[1];
	     temp2 *= (1.-diis_alpha);
	     auto newF = temp1;
	     newF += temp2;
	  }	  
       }
       else {
           auto is_diis = "DIIS";
	   auto dim = diis_dim;

	   if (dim == max_diis-1) { //Max amount stored/written
	       std::cout << " DIIS subspace collapsed\n";
	       diis_dim = 0;
           }

	   e_vecs[dim] = e;
	   auto B = einsums::Tensor<std::complex<double>, 2>("B matrix", dim+1, dim+1);
	   auto Y = einsums::Tensor<std::complex<double>, 2>("Y", 1, dim+1);
	   B.zero();
	   Y.zero();
           Y(0, dim) = {-1.0, 0.0};
           for (int i = 0; i < dim; i++) {
	       for (int j = 0; j < dim; j++) {
                   for (int k = 0; k < irrep_sizes_[irrep]; k++) {
		       for (int l = 0; l < irrep_sizes_[irrep]; l++) {
		           B(i, j) += e_vecs[i](k, l)*e_vecs[j](k, l);
		       }
		   }
	       }		   
	   }

	   for (int i = 0; i < dim; i++) {
		   B(i, dim) = {-1.0, 0.0};
		   B(dim, i) = {-1.0, 0.0};
	   }

	   //println(B);

	   diis_dim ++;

	   //auto C = einsums::Tensor<std::complex<double>, 1>("C", dim+1);
           auto IPIV = einsums::Tensor<std::complex<double>, 1>("IPIV", dim+1);

           auto C = einsums::linear_algebra::gesv(&B, &Y);
	   std::cout << C << "\n";
	   println(Y);

	   //auto newF = einsums::Tensor<std::complex<double, 2>("New F", irrep_sizes_[irrep], irrep_sizes_[irrep]);
	   //newF.zero();
	   for (int i = 0; i < irrep_sizes_[irrep]; i++) {
	       for (int j = 0; j < irrep_sizes_[irrep]; j++) {
		   //Fp_[irrep](i, j) = {0.0, 0.0};

	           for (int k = 0; k < dim; k++) {
                       //Fp_[irrep](i, j) += F_vecs[k][irrep](i, j)*Y(0, k);
		   
		   }		   
		   

		}
	   }
	   
	    
       }	       
    }
}

void GHF::SCF(Options& options, MintsHelper mints) {
    bool diis_selected = true;
	
    form_S();
    form_X();
    form_F();
    //form_G();
    int maxiter = 10;
    double scf_thresh = 1e-5;
    int max_diis = 6; //diis_max
    int total_diis = 0; //diis_count
    int e_rms = 1e6;
    double E = 0;
    DIIS diis;

    if (diis_selected) {
	//Should probably do some class inheritance here
	//but I'm much too dumb to figure that out in C++ right now
	diis.build_arrays(max_diis);
        diis.irrep_sizes_ = irrep_sizes_;
	diis.nirrep_ = nirrep_;
	diis.X_ = X_;
	diis.S_ = S_;
    }
    
    for (int i = 0; i < maxiter; i++) {
        F_.zero();
        F_ += F0_;
        F_ += J_;
        F_ -= K_;

	if (diis_selected && i+1 > 2) {
	    diis.do_diis(F_, D_, F1, Fp_);
	}
	else {
            orthogonalize_fock();
	}
            diagonalize_fock();
            back_transform();
            sort_real_evals();

            form_D();
            build_JK(options, mints);
            double newE = compute_energy(options, mints);
	    double diffE = std::abs(E-newE);
	    std::cout << E << " " << newE << "\n";
	    std::cout << "Iter " << i+1 << " | " << std::setprecision(15) << E+Enuc << " | deltaE = " << diffE << "\n";
	    E = newE;

	    if (diffE < scf_thresh) {
		//std::cout << "\nConverged!\n";

		//break;
	    }
	}
}

}} // End namespaces










