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

#ifndef _psi_src_lib_libmints_complexmatrix_h
#define _psi_src_lib_libmints_complexmatrix_h

#include "psi4/pragma.h"

#include <complex>
#include <memory>
#include <string>
#include <vector>
#include <array>

#ifdef USING_Einsums
#include <Einsums/Config.hpp>
#include <Einsums/Tensor.hpp>
#endif

namespace psi {

class PSIO;
class Dimension;

#ifdef USING_Einsums

/*! \ingroup MINTS
 *  \class ComplexMatrix
 *  \brief Complex blocked matrix backed by an einsums TiledTensor.
 *
 *  Wraps einmsums::TiledTensor<std::complex<double>, 2> as a private member
 *  and provides implicit conversion to/from the underlying TiledTensor so
 *  all existing TiledTensor API (tile, grid_size, has_tile, zero, name, …)
 *  remains directly available.  Adds operations needed by DIIS and the
 *  Python layer: clone, axpy, subtract, vector_dot, save, and load.
 */
class PSI_API ComplexMatrix {
  public:
    using BlockT = einsums::Tensor<std::complex<double>, 2>;
    using TiledT = einsums::TiledTensor<std::complex<double>, 2>;
    using ValueType = std::complex<double>;

    // -- constructors (forward to TiledTensor) --

    ComplexMatrix() = default;
    ComplexMatrix(const ComplexMatrix&) = default;
    ComplexMatrix(ComplexMatrix&&) = default;
    ComplexMatrix& operator=(const ComplexMatrix&) = default;
    ComplexMatrix& operator=(ComplexMatrix&&) = default;

    /// Construct with matching row/col tile sizes (square diagonal tiles).
    ComplexMatrix(const std::string& name, const std::vector<size_t>& tile_sizes)
        : tensor_(name, tile_sizes) {}

    ComplexMatrix(const std::string& name, const std::array<std::vector<int>, 2>& tile_sizes)
        : tensor_(name, tile_sizes) {}

    /// Construct with independent row/col tile sizes (rectangular diagonal tiles).
    ComplexMatrix(const std::string& name, const std::vector<size_t>& row_sizes,
                  const std::vector<size_t>& col_sizes)
        : tensor_(name, row_sizes, col_sizes) {}

    /// Overload for std::vector<int> callers (e.g. copy_matrix_to_complex).
    ComplexMatrix(const std::string& name, const std::vector<int>& row_sizes,
                  const std::vector<int>& col_sizes)
        : tensor_(name,
                  std::vector<size_t>(row_sizes.begin(), row_sizes.end()),
                  std::vector<size_t>(col_sizes.begin(), col_sizes.end())) {}

		/// Overload for single block of size
	  ComplexMatrix(int rows, int cols)
	      : tensor_("", std::vector<int>{ rows }, std::vector<int>{ cols }) {}

		/// Overload for single block of size with name
	  ComplexMatrix(const std::string& name, int rows, int cols)
	      : tensor_(name, std::vector<int>{ rows }, std::vector<int>{ cols }) {}

    /// Overload for Dimension callers
    ComplexMatrix(const std::string& name, const Dimension& row_sizes, const Dimension& col_sizes);

    // -- implicit conversion: all TiledTensor methods transparently available --

    operator TiledT&() { return tensor_; }
    operator const TiledT&() const { return tensor_; }

    // -- ComplexMatrix-specific operations --

    /// Deep copy.
    std::shared_ptr<ComplexMatrix> clone() const;

    /// In-place self += alpha * other (diagonal tiles only).
    void axpy(std::complex<double> alpha, const ComplexMatrix& other);

    /// In-place self -= other (diagonal tiles only).
    void subtract(const ComplexMatrix& other);

    /// Re(Tr(self^H other)), summed over diagonal tiles.
    double vector_dot(const ComplexMatrix& other) const;

    /// Save diagonal tiles as raw complex sub-blocks to a PSIO file.
    void save(std::shared_ptr<PSIO>& psio, size_t fileno);

    /// Load diagonal tiles as raw complex sub-blocks from a PSIO file.
    void load(std::shared_ptr<PSIO>& psio, size_t fileno);

    const std::string& name() const { return tensor_.name(); }

    /// python compat printer
    void print_out() const { print("outfile"); }
    /// Print to an ostream (delegates to the underlying TiledTensor).
    void print(std::string outfile = "outfile", const char* extra = nullptr) const;

    int rowdim(const int& h = 0) const { return tensor_.tile_size(0)[h]; }
    int coldim(const int& h = 0) const { return tensor_.tile_size(1)[h]; }

    const Dimension rowspi() const;
    int rowspi(const int& h) const { return rowdim(h); }
    const Dimension colspi() const;
    int colspi(const int& h) const { return coldim(h); }

	constexpr int nrow() const { return static_cast<int>(tensor_.dim(0)); }
	constexpr int ncol() const { return static_cast<int>(tensor_.dim(1)); }

    bool has_block(const int& h) const { return tensor_.has_tile(h, h); }

	const std::array<std::vector<int>, 2>& block_sizes() const { return tensor_.tile_sizes(); }

    void zero() { tensor_.zero(); }

	/// Getters
    BlockT& get(const int& h) { return tensor_.tile(h, h); }
    const BlockT& get(const int& h) const { return tensor_.tile(h, h); }

    ValueType get(const int& h, const int& i, const int& j) { return tensor_.tile(h, h)(i, j); }
    const ValueType get(const int& h, const int& i, const int& j) const { return tensor_.tile(h, h)(i, j); }

    BlockT& operator[](const int& h) { return tensor_.tile(h, h); }
    const BlockT& operator[](const int& h) const { return tensor_.tile(h, h); }

    BlockT& operator()(const int& h) { return tensor_.tile(h, h); }
    const BlockT& operator()(const int& h) const { return tensor_.tile(h, h); }

    ValueType operator()(const int& h, const int& i, const int& j) { return tensor_.tile(h, h)(i, j); }
    const ValueType operator()(const int& h, const int& i, const int& j) const { return tensor_.tile(h, h)(i, j); }

    /// Setters
    void set(const int& h, const int& i, const int& j, const ValueType& value) { tensor_.tile(h, h)(i, j) = value; }
  private:
    TiledT tensor_;
};

using SharedComplexMatrix = std::shared_ptr<ComplexMatrix>;

#else  // !USING_Einsums

/// Stub type so pybind can expose ComplexMatrix without Einsums.
class PSI_API ComplexMatrix {};
using SharedComplexMatrix = std::shared_ptr<ComplexMatrix>;

#endif  // USING_Einsums

}  // namespace psi

#endif
