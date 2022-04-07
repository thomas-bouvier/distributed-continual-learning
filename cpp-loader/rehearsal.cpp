#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stream_loader.hpp"
namespace py = pybind11;

PYBIND11_MODULE(rehearsal, m) {
    m.doc() = "pybind11 based streaming rehearsal buffer"; // optional module docstring
    py::class_<stream_loader_t>(m, "StreamLoader")
        .def(py::init<int, int>())
	.def("accumulate", &stream_loader_t::accumulate);
}
