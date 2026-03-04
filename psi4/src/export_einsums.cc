
#ifdef USING_Einsums

#include <Einsums/Config.hpp>
#include <Einsums/Tensor.hpp>
#include "psi4/pybind11.h"
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

using Tensor2cd = einsums::Tensor<std::complex<double>, 2>;
using BlockTensor2cd = einsums::BlockTensor<std::complex<double>, 2>;

using namespace pybind11::literals;

void export_einsums(py::module& m) {
    py::class_<Tensor2cd>(m, "Tensor2cd", "einsums::Tensor<std::complex<double>, 2>")
        .def("shape", [](const Tensor2cd &t) {
            auto s = t.dims();
            return py::make_tuple(s[0], s[1]);
        })
        .def("to_array", [](Tensor2cd &t) {
            auto s = t.dims();
            return py::array_t<std::complex<double>>(
                {s[0], s[1]}, // shape
                {sizeof(std::complex<double>) * s[1],
                 sizeof(std::complex<double>)},  // row-major strides
                t.data(),
                py::cast(&t) // keep parent alive
            );
        })
        .def("__setitem__", [](Tensor2cd& t, std::tuple<int, int> idx, std::complex<double> v) {
            t(std::get<0>(idx), std::get<1>(idx)) = v;
        },"idx"_a, "v"_a)
        .def("__getitem__", [](const Tensor2cd &t, std::tuple<int, int> idx) {
            return t.subscript(std::get<0>(idx), std::get<1>(idx));
        }, "idx"_a)
        .def("__repr__", [](const Tensor2cd &t) {
            std::ostringstream oss;
            einsums::fprintln(oss, t);
            return oss.str();
        });
    py::class_<BlockTensor2cd>(m, "BlockTensor2cd", "einsums::BlockTensor<std::complex<double>, 2>")
        .def("block", [](const BlockTensor2cd &t, int h) {
            return t.block(h);
        }, "h"_a)
        .def("__getitem__", [](const BlockTensor2cd &t, std::tuple<int, int, int> idx) {
            return t.block(std::get<0>(idx)).subscript(std::get<1>(idx), std::get<2>(idx));
        }, "idx"_a)
        .def("__repr__", [](const BlockTensor2cd &t) {
            std::ostringstream oss;
            einsums::fprintln(oss, t);
            return oss.str();
        });
}

#endif
