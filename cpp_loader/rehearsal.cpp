#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "stream_loader.hpp"
#include "distributed_stream_loader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rehearsal, m) {
    m.doc() = "pybind11 based streaming rehearsal buffer"; // optional module docstring

    py::class_<stream_loader_t>(m, "StreamLoader")
        .def(py::init<unsigned int, unsigned int, unsigned int, int64_t>(), py::call_guard<py::scoped_ostream_redirect>())
        .def("accumulate", &stream_loader_t::accumulate, py::call_guard<py::gil_scoped_release>())
        .def("wait", &stream_loader_t::wait, py::call_guard<py::gil_scoped_release>())
        .def("get_rehearsal_size", &stream_loader_t::get_rehearsal_size)
        .def("get_history_count", &stream_loader_t::get_history_count);

    py::class_<distributed_stream_loader_t>(m, "DistributedStreamLoader")
        .def(py::init<unsigned int, unsigned int, unsigned int, int64_t, uint16_t, std::string, std::vector<std::pair<int, std::string>>>(), py::call_guard<py::scoped_ostream_redirect>())
        .def("accumulate", &distributed_stream_loader_t::accumulate, py::call_guard<py::gil_scoped_release>())
        .def("wait", &distributed_stream_loader_t::wait, py::call_guard<py::gil_scoped_release>())
        .def("get_rehearsal_size", &distributed_stream_loader_t::get_rehearsal_size)
        .def("get_history_count", &distributed_stream_loader_t::get_history_count);
}
