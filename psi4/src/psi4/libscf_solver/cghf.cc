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
    Fa_ = SharedMatrix(factory_->create_matrix("F alpha"));
    Fb_ = SharedMatrix(factory_->create_matrix("F beta"));

    Gaa_ = SharedMatrix(factory_->create_matrix("F alpha"));
    Gab_ = SharedMatrix(factory_->create_matrix("F alpha"));
    Gba_ = SharedMatrix(factory_->create_matrix("F alpha"));
    Gbb_ = SharedMatrix(factory_->create_matrix("F alpha"));
    Va_ = SharedMatrix(factory_->create_matrix("V alpha"));
    Vb_ = SharedMatrix(factory_->create_matrix("V beta"));
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

    wKa_ = SharedMatrix(factory_->create_matrix("wK alpha"));
    wKb_ = SharedMatrix(factory_->create_matrix("wK beta"));
    //J_ = SharedMatrix(factory_->create_matrix("J total"));

    form_S();

    nelectron_ -= charge_;
    int nbocc_ = (nelectron_ - multiplicity_ + 1) / 2;
    //TODO fix nalphapi_ to be the actual nalphapi_ from HF
    //This is a bandage to the overarching problem that
    //we can't access it for some reason
    nalphapi_ = nbocc_ + multiplicity_ - 1;
    //auto nelpi_ = &nalphapi_;
    
    /*
    if (nirrep_ > 1) {
        for (int i = 0; i < nirrep_; i ++) {
            nelpi_[i] += nbetapi_[i];
        }
    }
    else {
         nelpi_ += nbetapi_; 
    }
    */

    REvecs_ = einsums::BlockTensor<std::complex<double>, 2>("R Eigenvectors", irrep_sizes_);
    LEvecs_ = einsums::BlockTensor<std::complex<double>, 2>("L Eigenvectors", irrep_sizes_);

    Evals_ = einsums::BlockTensor<std::complex<double>, 1>("Eigenvalues", irrep_sizes_);
    RealEvals_ = einsums::BlockTensor<double, 1>("Real Eigenvalues", irrep_sizes_);
    twoe_ = einsums::BlockTensor<std::complex<double>, 2>("DUMB", irrep_sizes_);

    F_ = einsums::BlockTensor<std::complex<double>, 2>("AO Fock", irrep_sizes_);
    EINT_ = einsums::BlockTensor<std::complex<double>, 2>("Kinetic Matrix", irrep_sizes_);
    Fp_ = einsums::BlockTensor<std::complex<double>, 2>("AO Fock Ortho", irrep_sizes_);
    JKwK_ = einsums::BlockTensor<std::complex<double>, 2>("JKwK_", irrep_sizes_);

    D_ = einsums::BlockTensor<std::complex<double>, 2>("AO Density", irrep_sizes_);
    C_ = einsums::BlockTensor<std::complex<double>, 2>("C", irrep_sizes_);
    Cocc_ = einsums::BlockTensor<std::complex<double>, 2>("Cocc", irrep_sizes_);
    //J_ = einsums::BlockTensor<std::complex<double>, 2>("J", irrep_sizes_);
    //K_ = einsums::BlockTensor<std::complex<double>, 2>("K", irrep_sizes_);
    /*
    Jaa_ = einsums::Tensor<std::complex<double>, 2>("Jaa", nsopi_[i], nsopi_[i]);
    Jbb_ = einsums::Tensor<std::complex<double>, 2>("Jbb", nsopi_[i], nsopi_[i]);

    Kaa_ = einsums::Tensor<std::complex<double>, 2>("Kaa", nsopi_[i], nsopi_[i]);
    Kab_ = einsums::Tensor<std::complex<double>, 2>("Kab", nsopi_[i], nsopi_[i]);
    Kba_ = einsums::Tensor<std::complex<double>, 2>("Kba", nsopi_[i], nsopi_[i]);
    Kbb_ = einsums::Tensor<std::complex<double>, 2>("Kbb", nsopi_[i], nsopi_[i]);
    */
    JKwK_.zero();
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
    //J_.zero();
    //K_.zero();
    twoe_.zero();
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
    
    //Fa_ = SharedMatrix(factory_->create_matrix("F alpha"));
    //Fb_ = SharedMatrix(factory_->create_matrix("F beta"));

    G_mat = mintshelper()->ao_eri();
    //T_ = mintshelper()->so_kinetic();
    //V_ = mintshelper()->so_potential();
    //T_->print("outfile");
    //V_->print("outfile");
    //auto molecule_ = basisset_->molecule();
    form_H();
    form_X();
    form_init_F();
    subclass_init();
    //SCF(options(), mints);
}

void CGHF::form_H() {
    T_ = mintshelper()->so_kinetic();
    V_ = mintshelper()->so_potential();

}


void CGHF::form_S() {
    //S_mat->set_name("Overlap Matrix");
    for (int i = 0; i < nirrep_; i++) {
        irrep_sizes_.push_back(2*S_->coldim(i));

        auto S_block = einsums::Tensor<double, 2>(
                        "S block", 2*Fa_->rowdim(i), 2*Fa_->coldim(i));
        S_block.zero();
        for (int j = 0; j < Fa_->rowdim(i); j++) {
                for (int k = 0; k < Fa_->coldim(i); k++) {
                        S_block(j, k) = S_->get(i, j, k);
                        S_block(j+Fa_->rowdim(i), k+Fa_->coldim(i)) = S_->get(i, j, k);
                }
        }
        EINS_.push_block(S_block);

    }
}


void CGHF::form_X() {
    //auto X_mat = einsums::linear_algebra::pow(S_, -0.5);
    //X_mat->print();
    //X_mat->power(-0.5, 1e-14);
    //X_mat->print();
    auto X_real = einsums::linear_algebra::pow(EINS_, -0.5, 1e-14);

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

void CGHF::form_V() {
 // Placeholder
}

void CGHF::form_init_F() {
    //F_mat->add(V_mat);
    //F_.zero();
    for (int i = 0; i < nirrep_; i++) {
        auto F_block = einsums::Tensor<std::complex<double>, 2>(
                      "F block", irrep_sizes_[i], irrep_sizes_[i]);

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

    //F_ += F0_;
    //F_ += J_;
    //F_ -= K_;
}

void CGHF::form_F() {
   F_.zero();
   F_ += F0_;
   F_ += JKwK_;
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
       auto Fp_irrep = Fp_[i];
       //auto Fp_irrep = Fp_blocks[i];
       if (irrep_sizes_[i] > 0) {
           einsums::linear_algebra::geev(&Fp_irrep, &Evals_[i], &LEvecs_[i], &REvecs_[i]);
       }
    //println(Fp_);
    //println(Evals_);


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
    nocc_.assign(nirrep_, 0.0);

    for (int i = 0; i < nirrep_; i++) {
            std::map<int, double> eval_map_temp;
            int orderpi [irrep_sizes_[i]];
            for (int j = 0; j < irrep_sizes_[i]; j++) {
                double eval = Eval_blocks[i](j);
		if (eval > 0.0) {
		   nocc_[i]++;
		}
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

void CGHF::form_C(double shift) {
    orthogonalize_fock();
    diagonalize_fock();
    back_transform();
    sort_real_evals();


}

void CGHF::form_D() {
    auto C_block = C_.vector_data();
    D_.zero();
    Cocc_.zero();

    auto nelec_ = nalphapi_ + nbetapi_;
    for (int i = 0; i < nirrep_; i++) {
        auto Cocc = einsums::Tensor<std::complex<double>, 2>("Cocc", irrep_sizes_[i], nelec_[i]);
        Cocc.zero();
        for (int j = 0; j < irrep_sizes_[i]; j++) {
                for (int k = 0; k < 2*nalphapi_[i]; k++) {
                        Cocc(j,k) = C_block[i](j, k);
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

void CGHF::form_G() {
    timer_on("CGHF form_G");
    std::cout << "form_G\n";
    if (functional_->needs_xc()) {
        form_V();
        Gaa_->copy(Va_);
        Gbb_->copy(Vb_);
    } else {
	Gaa_->zero();
	Gbb_->zero();
        Gab_->zero();
        Gba_->zero();

    }

    JKwK_.zero();

    auto nelecpi_ = nalphapi_ + nbetapi_;
    //nelecpi_.n()

    // Push the C matrix on
    std::vector<SharedMatrix>& Cl = jk_->C_left();
    std::vector<SharedMatrix>& Cr = jk_->C_right();

    auto Cai_subset = std::make_shared<Matrix>(nirrep_, nsopi_, nelecpi_);
    auto Car_subset = std::make_shared<Matrix>(nirrep_, nsopi_, nelecpi_);

    auto Cbi_subset = std::make_shared<Matrix>(nirrep_, nsopi_, nelecpi_);
    auto Cbr_subset = std::make_shared<Matrix>(nirrep_, nsopi_, nelecpi_);
   
    Cr.clear();
    Cl.clear();
    
    /*
    #############################################################################

    The J and K objects are as follows:  -> assume Einstein Summation

    J_uv = g_uvsl * D_sl
    K_uv = g_ulsv * D_sl

    where our Density matrix is the contraction,

    D_sl = C_si * C_li

    Since we're doing cGHF, the coefficient matrices will have real and imag parts,

    C_si * C_li = (C_si + iC_si^)(C_li + iC_li^)

    where the ^ indicates the 'real' part of the imaginary component.

    Foiling this out gives,

    C_si * C_li = (C_si * C_li) + i(C_si * C_li^) + i(C_si^ * C_li) - (C_si^ * C_li^)

    It is important to note that the first and last terms are completely REAL
    since i^2 = -1, and the second and third terms are completely IMAG


    #############################################################################

    Here are the keys to navigate this for the future confused Matt,

    Greek:
    u - mu
    v - nu
    s - sigma
    l - lambda

    Psi4 terms:
             g - 2D two-electron matrix
	    jk_ - JK object
             J - coulomb term from jk_
             K - exchange term from jk_
    Car_subset - REAL alpha components of the occupied coefficients matrix
    Cai_subset - IMAG alpha components of the occupied coefficients matrix
    Cbr_subset - REAL beta components of the occupied coefficients matrix
    Cbi_subset - IMAG beta components of the occupied coefficients matrix
            Cl -  LEFT component in density matrix contraction
	    Cr - RIGHT component in density matrix contraction
            
          Jaai - IMAG alpha/alpha components of J
          Jabi - IMAG alpha/beta  components of J
          Jbai - IMAG beta/alpha  components of J
          Jbbi - IMAG beta/beta   components of J

          Kaai - IMAG alpha/alpha components of K
          Kabi - IMAG alpha/beta  components of K
          Kbai - IMAG beta/alpha  components of K
          Kbbi - IMAG beta/beta   components of K

          Jaar - REAL alpha/alpha components of J
          Jabr - REAL alpha/beta  components of J
          Jbar - REAL beta/alpha  components of J
          Jbbr - REAL beta/beta   components of J

          Kaar - REAL alpha/alpha components of K
          Kabr - REAL alpha/beta  components of K
          Kbar - REAL beta/alpha  components of K
          Kbbr - REAL beta/beta   components of K


    Einsums terms (mycode):
             D - density matrix from Einsums
	  Cocc - occupied coefficient matrix
	 JKwK_ - BlockTensor with the combined J and K terms (and optional w percentage with DFT)

    #############################################################################

    */

    for (int i = 0; i < nirrep_; i++) {
        for (int j = 0; j < nsopi_[i]; j++) {
	    for (int k = 0; k < nelectron_; k++) {
		Car_subset->set(i, j, k, Cocc_[i](j, k).real());
                Cai_subset->set(i, j, k, Cocc_[i](j, k).imag());
                Cbr_subset->set(i, j, k, Cocc_[i](j+nsopi_[i], k).real());
                Cbi_subset->set(i, j, k, Cocc_[i](j+nsopi_[i], k).imag());
	        }
	    }
        }

    timer_on("Jaa/Kaa");
    Cl.push_back(Car_subset);
    Cl.push_back(Car_subset);
    Cl.push_back(Cai_subset);
    Cl.push_back(Cai_subset);

    Cr.push_back(Car_subset);
    Cr.push_back(Cai_subset);
    Cr.push_back(Car_subset);
    Cr.push_back(Cai_subset);

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
                JKwK_[i](p+nsopi_[i], q) -= std::complex<double>(Kabr_->get(i, p, q), Kabi_->get(i, p, q));
            }
        }
    }
    timer_off("CGHF form_G");
 
}

void CGHF::finalize() {
    F_.zero();
    F_ += F0_;
    //F_ += JKwK_;
    HF::finalize();
}


void CGHF::form_2e() {
   twoe_.zero();
   for (int i = 0; i < nirrep_; i++) {
       int row_dim = G_mat->rowdim(i);
       int col_dim = G_mat->coldim(i);
       auto G_block_super = einsums::Tensor<std::complex<double>, 2>("G block", row_dim, col_dim);
       G_block_super.zero();

       for (int j = 0; j < row_dim; j++) {
           for (int k = 0; k < col_dim; k++) {
	       G_block_super(j, k) = G_mat->get(i, j, k);
	   }
       twoe_.push_block(G_block_super);
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

void CGHF::form_T() {
    EINT_.zero();
    for (int i = 0; i < nirrep_; i++) {
        auto T_block = einsums::Tensor<double, 2>(
                        "T block", 2*T_->rowdim(i), 2*T_->coldim(i));

        T_block.zero();
        for (int j = 0; j < T_->rowdim(i); j++) {
                for (int k = 0; k < T_->coldim(i); k++) {
                        T_block(j, k) = T_->get(i, j, k);
                        T_block(j+T_->rowdim(i), k+T_->coldim(i)) = T_->get(i, j, k);
                }
        }
        EINT_[i] = T_block;

    //println(EINT_);

  }

}


void CGHF::compute_potential() {
  // Placeholder
}

double CGHF::compute_kinetic_E() {
    auto TensorKE = einsums::Tensor<std::complex<double>, 0>("KE");

    form_T();
    /*
    einsum(einsums::tensor_algebra::Indices{}, &TensorKE,
           einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i},
           EINT_,
           einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i},
           D_);

    std::complex<double> KE_complex = (std::complex<double>)TensorKE;
    double KE = KE_complex.real();
    */

    return 1.0;
}

double CGHF::compute_1e_E() {
	return 0.0;
}

double CGHF::compute_coulomb_E() {
	return 0.0;
}

//double CGHF::compute_energy(Options& options, MintsHelper mints) {
double CGHF::compute_E() {

    auto TensorE = einsums::Tensor<std::complex<double>, 0>("E");

    auto temp = einsums::BlockTensor<std::complex<double>, 2>("temp", irrep_sizes_);
    temp = F0_;
    for (int i = 0; i < nirrep_; i++) {
        auto JK_block = JKwK_[i];
        auto temp_JK_block = JK_block;
	
	temp_JK_block *= 0.5;

        temp[i] += temp_JK_block;
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

    //energies_["Nuclear"] = nuclearrep_;
    std::cout << "Kinetic Energy: " << compute_kinetic_E() << "\n";
    energies_["Kinetic"] = compute_kinetic_E();
    energies_["Total Energy"] = E+nuclearrep_;
    //energies_["One-Electron"] = one_electron_E;
    //energies_["Two-Electron"] = 0.5 * (coulomb_E + exchange_E);
    std::cout << "Energy " << E << "\n";
    std::cout << "Nuclear repulsion " << nuclearrep_ << "\n";
    return E+nuclearrep_;
}

/*
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
        //diagonalize_fock();
        //back_transform();
        //sort_real_evals();

        form_D();
        build_JK(options, mints);
	double newE = compute_energy(options, mints);
	double diffE = std::abs(E-newE);
	std::cout << E << " " << newE << "\n";
	std::cout << "Iter " << i+1 << " | " << std::setprecision(15) << E+Enuc << " | deltaE = " << diffE << "\n";
	E = newE;
        
	if (diffE < scf_thresh) {
	    //std::cout << "\nConverged!\n";

	    break;
	    }
	}
	
}
*/



}} // End namespaces










