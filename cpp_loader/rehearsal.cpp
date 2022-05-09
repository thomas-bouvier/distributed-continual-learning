#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stream_loader.hpp"
namespace py = pybind11;

PYBIND11_MODULE(rehearsal, m) {
    m.doc() = "pybind11 based streaming rehearsal buffer"; // optional module docstring
    py::class_<stream_loader_t>(m, "StreamLoader")
        .def(py::init<unsigned int, unsigned int, unsigned int, int64_t>())
    .def("accumulate", &stream_loader_t::accumulate, py::call_guard<py::gil_scoped_release>())
    .def("wait", &stream_loader_t::wait, py::call_guard<py::gil_scoped_release>())
    .def("get_rehearsal_size", &stream_loader_t::get_rehearsal_size);
}
