cmake_minimum_required(VERSION 3.10)
project(mocha)

include_directories(include)
aux_source_directory(src MOCHA_SRC)

find_package(LLVM CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(HandleLLVMOptions)
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs Support Core IRReader)

add_library(mocha ${MOCHA_SRC})
target_include_directories(mocha PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(mocha PRIVATE ${llvm_libs} 
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms
    mylib pybind11::lto pybind11::embed pybind11::module)