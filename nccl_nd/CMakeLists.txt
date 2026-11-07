# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

cmake_minimum_required(VERSION 3.20)
# Compiler detection must use an architecture available at the minimum toolkit.
set(_nd_default_architectures OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "90")
  set(_nd_default_architectures ON)
endif()
project(nccl_nd LANGUAGES CXX CUDA)
find_package(CUDAToolkit 12.5 REQUIRED)
if(_nd_default_architectures AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CMAKE_CUDA_ARCHITECTURES "90;100")
endif()
include(GNUInstallDirs)

set(NCCL_HOME "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/nccl/build" CACHE PATH
    "NCCL build or installation with Device API headers")
set(NCCL_ND_BUILDDIR "${CMAKE_BINARY_DIR}" CACHE PATH "NCCL ND output directory")
if(NOT EXISTS "${NCCL_HOME}/include/nccl_device.h")
  message(FATAL_ERROR "NCCL_HOME must contain include/nccl_device.h; build the vendored NCCL first")
endif()
find_library(NCCL_LIB NAMES nccl HINTS "${NCCL_HOME}/lib" "${NCCL_HOME}/lib64" NO_DEFAULT_PATH)
if(NOT NCCL_LIB)
  file(GLOB _nccl_sonames "${NCCL_HOME}/lib/libnccl.so.*" "${NCCL_HOME}/lib64/libnccl.so.*")
  if(_nccl_sonames)
    list(GET _nccl_sonames 0 NCCL_LIB)
  endif()
endif()
if(NOT NCCL_LIB OR NOT EXISTS "${NCCL_LIB}")
  message(FATAL_ERROR "NCCL_HOME must contain libnccl.so or a versioned SONAME under lib/ or lib64/")
endif()
get_filename_component(_nccl_libdir "${NCCL_LIB}" DIRECTORY)
get_filename_component(_nccl_libname "${NCCL_LIB}" NAME)

file(READ include/nccl_nd.h _nd_header)
foreach(part MAJOR MINOR PATCH)
  string(REGEX MATCH "#define NCCL_ND_${part} ([0-9]+)" _match "${_nd_header}")
  set(NCCL_ND_${part} "${CMAKE_MATCH_1}")
endforeach()
set(_nd_include "${NCCL_ND_BUILDDIR}/include")
set(_nd_lib "${NCCL_ND_BUILDDIR}/lib")
file(MAKE_DIRECTORY "${_nd_include}")
configure_file(include/nccl_nd.h "${_nd_include}/nccl_nd.h" COPYONLY)

add_library(nccl_nd SHARED nccl_nd.cu)
set_target_properties(nccl_nd PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_nd_lib}"
    VERSION "${NCCL_ND_MAJOR}.${NCCL_ND_MINOR}.${NCCL_ND_PATCH}"
    SOVERSION "${NCCL_ND_MAJOR}"
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
    BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
target_include_directories(nccl_nd PUBLIC
    $<BUILD_INTERFACE:${_nd_include}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(nccl_nd SYSTEM PUBLIC "${NCCL_HOME}/include")
foreach(cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
  target_include_directories(nccl_nd SYSTEM PRIVATE "${cuda_include}/cccl")
endforeach()
target_compile_options(nccl_nd PRIVATE
    --expt-relaxed-constexpr --extended-lambda
    -Xcompiler=-fvisibility=hidden,-Wall,-Wextra)
# nvcc treats bare versioned .so paths as input files; link the exact filename.
target_link_directories(nccl_nd PUBLIC "${_nccl_libdir}")
target_link_libraries(nccl_nd PUBLIC "-l:${_nccl_libname}" CUDA::cudart)
set_property(TARGET nccl_nd APPEND PROPERTY LINK_DEPENDS "${NCCL_LIB}")
target_link_options(nccl_nd PRIVATE "LINKER:-z,defs")

install(TARGETS nccl_nd LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES include/nccl_nd.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

option(NCCL_ND_BUILD_TESTS "Build nd_test and nd_fault_test" OFF)
if(NCCL_ND_BUILD_TESTS)
  set(_nd_test_dir "${NCCL_ND_BUILDDIR}/test/nccl_nd")
  add_executable(nd_test tests/nd_test.cu)
  set_target_properties(nd_test PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${_nd_test_dir}"
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_link_libraries(nd_test PRIVATE nccl_nd)
  # Compiles the library source with test-only fault hooks; does not link nccl_nd.
  add_executable(nd_fault_test tests/nd_fault_test.cu)
  set_target_properties(nd_fault_test PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${_nd_test_dir}"
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_include_directories(nd_fault_test PRIVATE "${_nd_include}")
  target_include_directories(nd_fault_test SYSTEM PRIVATE "${NCCL_HOME}/include")
  foreach(cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
    target_include_directories(nd_fault_test SYSTEM PRIVATE "${cuda_include}/cccl")
  endforeach()
  target_compile_options(nd_fault_test PRIVATE --expt-relaxed-constexpr)
  target_link_directories(nd_fault_test PRIVATE "${_nccl_libdir}")
  target_link_libraries(nd_fault_test PRIVATE "-l:${_nccl_libname}" CUDA::cudart)
endif()
