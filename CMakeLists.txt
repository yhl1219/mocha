cmake_minimum_required(VERSION 3.10)
project(mocha_mlir)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
aux_source_directory(src MOCHA_SRC)

find_package(LLVM CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(HandleLLVMOptions)
add_definitions(${LLVM_DEFINITIONS})

find_package(MLIR CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)

pybind11_add_module(mocha_mlir ${MOCHA_SRC})
target_include_directories(mocha_mlir PRIVATE ${LLVM_INCLUDE_DIRS})
llvm_map_components_to_libnames(llvm_libs Support Core IRReader)
target_include_directories(mocha_mlir PRIVATE ${MLIR_INCLUDE_DIRS})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mocha_mlir PRIVATE
    ${llvm_libs}
    ${dialect_libs}
    MLIRAnalysis MLIRIR MLIRParser MLIRSideEffectInterfaces MLIRTransforms MLIRPass MLIRStandard 
    pybind11::lto pybind11::embed pybind11::module
    jsoncpp_lib jsoncpp_object)