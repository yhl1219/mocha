#include <stdexcept>
//
#include <pybind11/pybind11.h>

PYBIND11_MODULE(mochapp, m) {
  m.doc() = "mocha c++ layer binding";
  m.def("enter_builder", &create_builder,
        "create a mocha MLIR context builder");
  m.def("exit_builder", &exit_builder, "close the mocha MLIR context builder");
  
}

int main(int argc, char **argv) { return 0; }