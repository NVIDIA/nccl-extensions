cmake_minimum_required(VERSION 3.25)

# NCCL EP Example/Library CMake Configuration

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(NCCL_EP LANGUAGES CUDA CXX)
endif()

include(GNUInstallDirs)

if(NOT CUDAToolkit_FOUND)
  find_package(CUDAToolkit REQUIRED)
endif()

if(NOT DEFINED ENV{CUDA_HOME} OR "$ENV{CUDA_HOME}" STREQUAL "")
  if(DEFINED CUDAToolkit_ROOT AND NOT "${CUDAToolkit_ROOT}" STREQUAL "")
    set(ENV{CUDA_HOME} "${CUDAToolkit_ROOT}")
  elseif(DEFINED CUDAToolkit_TARGET_DIR AND NOT "${CUDAToolkit_TARGET_DIR}" STREQUAL "")
    set(ENV{CUDA_HOME} "${CUDAToolkit_TARGET_DIR}")
  elseif(DEFINED CUDAToolkit_BIN_DIR AND NOT "${CUDAToolkit_BIN_DIR}" STREQUAL "")
    get_filename_component(_nccl_ep_cuda_home "${CUDAToolkit_BIN_DIR}" DIRECTORY)
    set(ENV{CUDA_HOME} "${_nccl_ep_cuda_home}")
  endif()
endif()

set(NCCL_EP_REPO_ROOT_CANDIDATES
  "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
)
foreach(_nccl_ep_repo_root_candidate IN LISTS NCCL_EP_REPO_ROOT_CANDIDATES)
  get_filename_component(_nccl_ep_repo_root "${_nccl_ep_repo_root_candidate}" ABSOLUTE)
  if(EXISTS "${_nccl_ep_repo_root}/build/include/nccl.h" AND EXISTS "${_nccl_ep_repo_root}/build/lib")
    set(NCCL_EP_REPO_ROOT "${_nccl_ep_repo_root}")
    break()
  endif()
endforeach()

if(NOT DEFINED ENV{NCCL_HOME} OR "$ENV{NCCL_HOME}" STREQUAL "")
  if(DEFINED NCCL_EP_REPO_ROOT)
    set(ENV{NCCL_HOME} "${NCCL_EP_REPO_ROOT}/build")
  endif()
endif()

if(NOT DEFINED ENV{NCCL_INCLUDE_DIR} OR "$ENV{NCCL_INCLUDE_DIR}" STREQUAL "")
  if(DEFINED ENV{NCCL_HOME} AND EXISTS "$ENV{NCCL_HOME}/include/nccl.h")
    set(ENV{NCCL_INCLUDE_DIR} "$ENV{NCCL_HOME}/include")
  elseif(DEFINED NCCL_EP_REPO_ROOT AND EXISTS "${NCCL_EP_REPO_ROOT}/build/include/nccl.h")
    set(ENV{NCCL_INCLUDE_DIR} "${NCCL_EP_REPO_ROOT}/build/include")
  endif()
endif()

if(NOT DEFINED ENV{NCCL_LIB_DIR} OR "$ENV{NCCL_LIB_DIR}" STREQUAL "")
  if(DEFINED ENV{NCCL_HOME} AND EXISTS "$ENV{NCCL_HOME}/lib")
    set(ENV{NCCL_LIB_DIR} "$ENV{NCCL_HOME}/lib")
  elseif(DEFINED NCCL_EP_REPO_ROOT AND EXISTS "${NCCL_EP_REPO_ROOT}/build/lib")
    set(ENV{NCCL_LIB_DIR} "${NCCL_EP_REPO_ROOT}/build/lib")
  endif()
endif()

set(NCCL_INCLUDE_DIR "$ENV{NCCL_INCLUDE_DIR}" CACHE PATH "Directory containing nccl.h and nccl_device.h")
set(NCCL_LIB_DIR "$ENV{NCCL_LIB_DIR}" CACHE PATH "Directory containing libnccl")

if(NOT NCCL_INCLUDE_DIR OR NOT EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
  message(FATAL_ERROR "NCCL_INCLUDE_DIR must point to NCCL headers containing nccl.h")
endif()
if(NOT EXISTS "${NCCL_INCLUDE_DIR}/nccl_device.h")
  message(FATAL_ERROR "NCCL_INCLUDE_DIR must point to NCCL headers containing nccl_device.h")
endif()

if(NCCL_LIB_DIR)
  find_library(NCCL_LIBRARY NAMES nccl PATHS "${NCCL_LIB_DIR}" NO_DEFAULT_PATH)
  if(NOT NCCL_LIBRARY)
    file(GLOB _nccl_ep_nccl_lib_candidates
      "${NCCL_LIB_DIR}/libnccl.so"
      "${NCCL_LIB_DIR}/libnccl.so.*"
      "${NCCL_LIB_DIR}/libnccl.a")
    if(_nccl_ep_nccl_lib_candidates)
      list(SORT _nccl_ep_nccl_lib_candidates)
      list(GET _nccl_ep_nccl_lib_candidates 0 NCCL_LIBRARY)
    endif()
  endif()
endif()

if(NOT NCCL_LIBRARY)
  find_library(NCCL_LIBRARY NAMES nccl)
endif()
if(NOT NCCL_LIBRARY)
  message(FATAL_ERROR "Could not find libnccl. Set NCCL_LIB_DIR or NCCL_HOME.")
endif()

set(NCCL_EP_NCCL_INCLUDE_DIRS
    ${NCCL_INCLUDE_DIR}
)
if(EXISTS "${NCCL_INCLUDE_DIR}/nccl_device")
  list(APPEND NCCL_EP_NCCL_INCLUDE_DIRS "${NCCL_INCLUDE_DIR}/nccl_device")
endif()

set(NCCL_EP_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
foreach(_nccl_ep_cuda_include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
  if(EXISTS "${_nccl_ep_cuda_include_dir}/cccl")
    list(APPEND NCCL_EP_CUDA_INCLUDE_DIRS "${_nccl_ep_cuda_include_dir}/cccl")
  endif()
endforeach()

set(NCCL_EP_LINK_LIBRARIES ${NCCL_LIBRARY})
if(TARGET CUDA::cudart)
  list(APPEND NCCL_EP_LINK_LIBRARIES CUDA::cudart)
elseif(CUDAToolkit_LIBRARIES)
  list(APPEND NCCL_EP_LINK_LIBRARIES ${CUDAToolkit_LIBRARIES})
endif()
if(TARGET CUDA::cuda_driver)
  list(APPEND NCCL_EP_LINK_LIBRARIES CUDA::cuda_driver)
endif()

if(DEFINED CMAKE_CUDA_ARCHITECTURES)
  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch LESS 90)
      message(FATAL_ERROR "NCCL_EP requires sm_90 or higher. CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    endif()
  endforeach()
endif()

# Source files
set(NCCL_EP_HOST_SRC
    nccl_ep.cc
)

set(NCCL_EP_DEVICE_SRC
    device/low_latency.cu
    device/hybridep_adapter.cu
)

set(NCCL_EP_SRC
    ${NCCL_EP_HOST_SRC}
    ${NCCL_EP_DEVICE_SRC}
)

add_library(nccl_ep_shared SHARED
    ${NCCL_EP_SRC}
)
add_library(nccl_ep_static STATIC
    ${NCCL_EP_SRC}
)
add_custom_target(nccl_ep_lib DEPENDS nccl_ep_shared nccl_ep_static)

set(NCCL_EP_TARGETS nccl_ep_shared nccl_ep_static)

set_target_properties(nccl_ep_shared nccl_ep_static PROPERTIES
    OUTPUT_NAME "nccl_ep"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "90"
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

set_target_properties(nccl_ep_shared PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "$ORIGIN"
)

# Configurable LSA team size range for compile-time instantiation.
# Restricting the range reduces build time during development.  Example:
#   cmake ... -D_NCCL_EP_LSA_TEAM_SIZE_MIN=8 -D_NCCL_EP_LSA_TEAM_SIZE_MAX=8
set(_NCCL_EP_LSA_TEAM_SIZE_MIN "4" CACHE STRING
    "Minimum LSA team size to compile (must be multiple of 4, min 4)")
set(_NCCL_EP_LSA_TEAM_SIZE_MAX "32" CACHE STRING
    "Maximum LSA team size to compile (must be multiple of 4, max 32)")
set(_NCCL_EP_NUM_LSA_TEAMS_LIST "1;2;3;4;8" CACHE STRING
    "List of NUM_LSA_TEAMS values to compile (subset of 1 2 3 4 8)")

set(NCCL_EP_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${NCCL_EP_NCCL_INCLUDE_DIRS}
    ${NCCL_EP_CUDA_INCLUDE_DIRS}
)

# Allow device code to call host constexpr with --expt-relaxed-constexpr
# Hides all symbols by default in the host compiled object files with -fvisibility=hidden
foreach(t IN LISTS NCCL_EP_TARGETS)
    target_compile_options(${t} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=default>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fvisibility=hidden>
    )
    target_compile_definitions(${t} PRIVATE
        _NCCL_EP_LSA_TEAM_SIZE_MIN=${_NCCL_EP_LSA_TEAM_SIZE_MIN}
        _NCCL_EP_LSA_TEAM_SIZE_MAX=${_NCCL_EP_LSA_TEAM_SIZE_MAX})
    target_include_directories(${t} PUBLIC ${NCCL_EP_INCLUDE_DIRS})
    target_link_libraries(${t} PUBLIC ${NCCL_EP_LINK_LIBRARIES})
endforeach()

foreach(N IN LISTS _NCCL_EP_NUM_LSA_TEAMS_LIST)
    foreach(t IN LISTS NCCL_EP_TARGETS)
        target_compile_definitions(${t} PRIVATE _NCCL_EP_NUM_LSA_TEAMS_${N}=1)
    endforeach()
endforeach()

# Copy headers to build/include (runs at configure)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
configure_file(include/nccl_ep.h    "${CMAKE_BINARY_DIR}/include/nccl_ep.h"    COPYONLY)
configure_file(include/ep_enums.h   "${CMAKE_BINARY_DIR}/include/ep_enums.h"   COPYONLY)

# Install to custom prefix
install(FILES
    include/nccl_ep.h
    include/ep_enums.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS ${NCCL_EP_TARGETS}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
