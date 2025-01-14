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

