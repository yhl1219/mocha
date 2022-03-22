#include <stdexcept>
//
#include "mc_builder.hh"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mocha_mlir, m) {
  m.doc() = "mocha c++ layer binding";
  py::class_<mocha::MochaGenerator>(m, "MochaGenerator")
      .def(py::init<const std::string &, const std::string &>());
}