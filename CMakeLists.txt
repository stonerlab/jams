cmake_minimum_required(VERSION 3.10...3.28 FATAL_ERROR)
project(jams LANGUAGES C CXX)

include(cmake/MessageQuiet.cmake)
include(cmake/Utils.cmake)

# Do not allow users to build the project inside of the source directory
ensure_out_of_source_build()

# Try to detect nvcc (the CUDA compiler). If it's found, enable CUDA support
# by default. If not found then disable CUDA support by default.
# JAMS_BUILD_CUDA can still be overridden when calling cmake.
find_program(NVCC "nvcc")
if (NVCC)
    option(JAMS_BUILD_CUDA "Build JAMS with CUDA support" ON)
else()
    option(JAMS_BUILD_CUDA "Build JAMS with CUDA support" OFF)
endif()

option(JAMS_BUILD_OFFLINE "Build JAMS without updating external dependencies online" OFF)
option(JAMS_BUILD_OMP "Build JAMS with OpenMP support" OFF)
option(JAMS_BUILD_FASTMATH "Build JAMS with fast math flags" ON)
option(JAMS_BUILD_MIXED_PREC "Build parts of JAMS using mixed precision." OFF)
option(JAMS_BUILD_TESTS "Build all of JAMS's unit tests." OFF)
option(JAMS_BUILD_DOCS "Build all of JAMS's documentation." OFF)
option(JAMS_BUILD_IN_DEPENDENCIES "Build JAMS dependencies in the same cmake project" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (JAMS_BUILD_CUDA)
    if(DEFINED ENV{CUDA_ARCHITECTURES} AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCHITECTURES})
    else()
        # Default architectures if the environment variable is not set
        set(CMAKE_CUDA_ARCHITECTURES "60;70;80;89;90")
    endif()

    set(MESSAGE_QUIET)
    enable_language(CUDA)
    unset(MESSAGE_QUIET)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()


if (${JAMS_BUILD_OFFLINE})
    set_property(DIRECTORY ${jams_SOURCE_DIR} PROPERTY EP_UPDATE_DISCONNECTED 1)
endif()

# Minimum versions required for external dependencies
set(JAMS_LIBCONFIG_VERSION_MIN "1.6")
set(JAMS_HDF5_VERSION_MIN "1.10")
set(JAMS_CUDA_VERSION_MIN "9.0")

# If JAMS_BUILD_IN_DEPENDENCIES is ON then we will download these tagged
# versions from github and build them here rather than linking external
# libraries.
if (JAMS_BUILD_IN_DEPENDENCIES)
    set(JAMS_LIBCONFIG_VERSION v1.8.1)  # https://github.com/hyperrealm/libconfig
    set(JAMS_HIGHFIVE_VERSION v2.4.0)   # https://github.com/highfive-devs/highfive
    set(JAMS_SPGLIB_VERSION v1.16.5)    # https://github.com/spglib/spglib
    set(JAMS_PCG_VERSION v0.98.1)       # https://github.com/imneme/pcg-cpp
endif()

# It's recommended that googletest is always built into projects rather that
# as an external library. Google call their master branch 'main'.
set(JAMS_GTEST_VERSION main)      # https://github.com/google/googletest.git

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(cmake/Dependencies.cmake)
include(cmake/Generate.cmake)

add_subdirectory(src/jams)

if (JAMS_BUILD_DOCS)
    add_subdirectory(docs/)
endif()

if (JAMS_BUILD_TESTS)
    add_subdirectory(src/jams/test)
endif()

message("-----------------------------------------------------------")
message("CMAKE CONFIGURATION COMPLETE")
message("-----------------------------------------------------------")
message("jams build options: ")
message(STATUS "JAMS_VERSION:                   " ${JAMS_VERSION})
message(STATUS "JAMS_BUILD_CUDA:                " ${JAMS_BUILD_CUDA})
message(STATUS "JAMS_BUILD_OMP:                 " ${JAMS_BUILD_OMP})
message(STATUS "JAMS_BUILD_FASTMATH:            " ${JAMS_BUILD_FASTMATH})
message(STATUS "JAMS_BUILD_MIXED_PREC:          " ${JAMS_BUILD_MIXED_PREC})
message(STATUS "JAMS_BUILD_TESTS:               " ${JAMS_BUILD_TESTS})
message(STATUS "JAMS_BUILD_DOCS:                " ${JAMS_BUILD_MIXED_PREC})
message(STATUS "JAMS_BUILD_OFFLINE:             " ${JAMS_BUILD_OFFLINE})
message(STATUS "JAMS_BUILD_IN_DEPENDENCIES:     " ${JAMS_BUILD_IN_DEPENDENCIES})

message("compiler options:")
message(STATUS "CMAKE_BUILD_TYPE:               " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_CXX_COMPILER:             " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_LINKER:                   " ${CMAKE_LINKER})
message(STATUS "CMAKE_MAKE_PROGRAM:             " ${CMAKE_MAKE_PROGRAM})
message(STATUS "CMAKE_CXX_COMPILE_FLAGS:        " ${CMAKE_CXX_COMPILE_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS:                " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS_DEBUG:          " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "CMAKE_CXX_FLAGS_RELEASE:        " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
message(STATUS "JAMS_FAST_MATH_OPT:             " ${JAMS_FAST_MATH_OPT})

if(JAMS_BUILD_CUDA)
    message("cuda options:")
    message(STATUS "CUDA_VERSION:                   " ${CUDA_VERSION})
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR:          " ${CUDA_TOOLKIT_ROOT_DIR})
    message(STATUS "CUDA_CUDART_LIBRARY:            " ${CUDA_CUDART_LIBRARY})
    message(STATUS "CUDA_HOST_COMPILER:             " ${CUDA_HOST_COMPILER})
    message(STATUS "CUDA_NVCC_EXECUTABLE:           " ${CUDA_NVCC_EXECUTABLE})
    message(STATUS "CMAKE_CUDA_ARCHITECTURES:       " ${CMAKE_CUDA_ARCHITECTURES})
endif()

message("-----------------------------------------------------------")
