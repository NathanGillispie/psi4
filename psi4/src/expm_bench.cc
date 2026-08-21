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

// Standalone benchmark for psi::linalg::expm on a random Hermitian matrix.
// No Python, no driver -- a plain native executable suitable for perf/strace.

#include "psi4/libmints/complexmatrix.h"

#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <Einsums/Runtime.hpp>

int main() {
    const char* ein_argv[4] = {"psi4", "--einsums:no-profiler-report", "--einsums:log-level", "3"};
    einsums::initialize(4, ein_argv);

    const int n = 2000;
    const int trials = 10;

    psi::ComplexMatrix A{"bench", n, n};

    // Fill a random Hermitian matrix: real diagonal, upper triangle random,
    // lower triangle = conjugate transpose of upper.
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < n; ++i) {
        A.set(0, i, i, std::complex<double>{dist(rng), 0.0});
        for (int j = i + 1; j < n; ++j) {
            std::complex<double> z{dist(rng), dist(rng)};
            A.set(0, i, j, z);
            A.set(0, j, i, std::conj(z));
        }
    }

    double checksum = 0.0;
    std::vector<double> times_ms;
    times_ms.reserve(trials);

    for (int t = 0; t < trials; ++t) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = psi::linalg::expm(A);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        times_ms.push_back(elapsed.count());
        checksum += std::abs(result->get(0, 0, 0));
    }

    double mean = 0.0;
    for (double v : times_ms) mean += v;
    mean /= static_cast<double>(trials);

    double variance = 0.0;
    for (double v : times_ms) variance += (v - mean) * (v - mean);
    variance /= static_cast<double>(trials - 1);  // sample standard deviation
    double stddev = std::sqrt(variance);

    std::cout << "expm benchmark: " << n << "x" << n << " random Hermitian, " << trials << " trials\n";
    std::cout << "mean   = " << mean << " ms\n";
    std::cout << "stddev = " << stddev << " ms\n";
    std::cout << "checksum = " << checksum << "\n";

    return 0;
}
