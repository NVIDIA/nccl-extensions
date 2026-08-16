# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# Private CI unit tests for the quantized-tensor prototype (issue #7).

set(CI_UNIT_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/scale_plane_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/quantize_test.cc")

add_executable(unit_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests_main.cc
    ${CI_UNIT_TEST_SRCS}
    ${NCCL_M2N_SRC_FULL_PATHS}
)
target_include_directories(unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${NCCL_M2N_SOURCE_DIR}/src
)
target_include_directories(unit_tests SYSTEM PRIVATE
    ${NCCL_M2N_SOURCE_DIR}/third_party
    ${_NCCL_M2N_NCCL_INCLUDES}
    ${CUDAToolkit_INCLUDE_DIRS}
)
if(MPI_FOUND)
  target_include_directories(unit_tests SYSTEM PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endif()
target_compile_definitions(unit_tests PRIVATE NCCL_OS_LINUX NCCL_M2N_TESTING)
target_compile_options(unit_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_link_libraries(unit_tests PRIVATE
    ${_NCCL_M2N_NCCL_TARGET}
    CUDA::cudart
    CUDA::cuda_driver
    ${_NCCL_M2N_GTEST_TARGET}
    Threads::Threads
)
set_target_properties(unit_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${NCCL_M2N_RUNTIME_DIR}"
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    BUILD_RPATH "${NCCL_M2N_LIBRARY_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib"
)
