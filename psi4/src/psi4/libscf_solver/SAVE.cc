/*
 * @BEGIN LICENSE
 *
 * CGHF by Matthew Ward, a plugin to:
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

#include "cghf.h"

using namespace einsums;
using namespace einsums::tensor_algebra;

namespace psi{
namespace scf {

CGHF::CGHF(SharedWavefunction ref_wfn, std::shared_ptr<SuperFunctional> func)
    : HF(ref_wfn, func, Process::environment.options, PSIO::shared_object()) {
    common_init();
}

CGHF::CGHF(SharedWavefunction ref_wfn, std::shared_ptr<SuperFunctional> func, Options& options,
         std::shared_ptr<PSIO> psio)
    : HF(ref_wfn, func, options, psio) {
    common_init();
}

CGHF::~CGHF() {}

void CGHF::common_init(){
    //F_mat  = T_->clone();
    //F_mat->add(V_);

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

    //TODO make general epsilon_
    epsilon_a_ = SharedVector(factory_->create_vector());
    epsilon_a_->set_name("placeholder alpha orbital energies");
    epsilon_b_ = SharedVector(factory_->create_vector());
    epsilon_b_->set_name("placeholder betaorbital energies");

    Ca_ = SharedMatrix(factory_->create_matrix("alpha MO coefficients (C)"));
    Cb_ = SharedMatrix(factory_->create_matrix("beta MO coefficients (C)"));

    Da_ = SharedMatrix(factory_->create_matrix("SCF density"));
    Db_ = Da_;

    subclass_init();
    //SCF(options(), mints);
}

void CGHF::form_F() {
    //std::cout << "Forming F\n";
    //F_mat->add(V_mat);

    for (int i = 0; i < nirrep_; i++) {
        auto F_block = einsums::Tensor<std::complex<double>, 2>(
                      molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]);

        F_block.zero();
        for (int j = 0; j < T_->rowdim(i); j++) {
            for (int k = 0; k < T_->coldim(i); k++) {
		    auto fjk = T_->get(i, j, k) + V_->get(i, j, k);
                    F_block(j,k) = fjk;
                    F_block(j+T_->rowdim(i), k+T_->coldim(i)) = fjk;
            }
        }
          F0_[i] = F_block;
    }
}

void CGHF::orthogonalize_fock() {
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

void CGHF::diagonalize_fock() {
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

void CGHF::back_transform() {
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

void CGHF::sort_real_evals(){
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

void CGHF::form_D() {
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

auto CGHF::MakeComplexBlock(auto A) {
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

void CGHF::build_JK(Options& options, MintsHelper mints) {
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


void CGHF::form_G() {
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

void CGHF::save_density_and_energy() {
 //Placeholder
}

void CGHF::setup_potential() {
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


void CGHF::evals_sanity_check() {
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

double CGHF::compute_energy(Options& options, MintsHelper mints) {
    //double CGHF::compute_E() {

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


void CGHF::SCF(Options& options, MintsHelper mints) {
	
    form_S();
    form_X();
    form_F();
    //form_G();
    int maxiter = 10;
    double scf_thresh = 1e-5;
    int e_rms = 1e6;
    double E = 0;

    for (int i = 0; i < maxiter; i++) {
        F_.zero();
        F_ += F0_;
        F_ += J_;
        F_ -= K_;

        orthogonalize_fock();
        diagonalize_fock();
        back_transform();
        sort_real_evals();

        form_D();
        build_JK(options, mints);
        /*
	double newE = compute_energy(options, mints);
	double diffE = std::abs(E-newE);
	std::cout << E << " " << newE << "\n";
	std::cout << "Iter " << i+1 << " | " << std::setprecision(15) << E+Enuc << " | deltaE = " << diffE << "\n";
	E = newE;
        
	if (diffE < scf_thresh) {
	    //std::cout << "\nConverged!\n";

	    break;
	    }
	*/
	}
	
}



}} // End namespaces










