cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(${CMAKE_VERSION} VERSION_GREATER "3.12.0")
    cmake_policy(SET CMP0074 NEW) # ``find_package()`` uses ``<PackageName>_ROOT`` variables.
endif()

function(message)
    if ((NOT MESSAGE_QUIET) OR (DEFINED VERBOSE))
        _message(${ARGN})
    endif()
endfunction()

include(cmake/Utils.cmake)
ensure_out_of_source_build()

option(JAMS_BUILD_OFFLINE "Build JAMS without updating external dependencies online" OFF)
option(JAMS_BUILD_OMP "Build JAMS with OpenMP support" OFF)
option(JAMS_BUILD_CUDA "Build JAMS with CUDA support" ON)
option(JAMS_BUILD_FASTMATH "Build JAMS with fast math flags" ON)
option(JAMS_BUILD_MIXED_PREC "Build parts of JAMS using mixed precision." OFF)
option(JAMS_BUILD_TESTS "Build all of JAMS's unit tests." OFF)
option(JAMS_BUILD_DOCS "Build all of JAMS's documentation." OFF)

set(MESSAGE_QUIET ON)
project(jams LANGUAGES C CXX)

if (JAMS_BUILD_CUDA)
    enable_language(CUDA)
endif()

unset(MESSAGE_QUIET)

if (${JAMS_BUILD_OFFLINE})
    set_property(DIRECTORY ${jams_SOURCE_DIR} PROPERTY EP_UPDATE_DISCONNECTED 1)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(VERSION_MAJOR "2" )
set(VERSION_MINOR "8" )
set(VERSION_PATCH "0" )
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# Latest libconfig version is 1.7.2 but the cmake support is broken. This is fixed in the current
# github master branch

set(JAMS_LIBCONFIG_VERSION f53e5de) # https://github.com/hyperrealm/libconfig
set(JAMS_HIGHFIVE_VERSION v2.1.1)   # https://github.com/BlueBrain/HighFive
set(JAMS_SPGLIB_VERSION v1.16.0)    # https://github.com/spglib/spglib
set(JAMS_PCG_VERSION v0.98.1)       # https://github.com/imneme/pcg-cpp
set(JAMS_GTEST_VERSION master)      # https://github.com/google/googletest.git

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
message(STATUS "VERSION:               " ${VERSION})
message(STATUS "JAMS_BUILD_CUDA:       " ${JAMS_BUILD_CUDA})
message(STATUS "JAMS_BUILD_OMP:        " ${JAMS_BUILD_OMP})
message(STATUS "JAMS_BUILD_FASTMATH:   " ${JAMS_BUILD_FASTMATH})
message(STATUS "JAMS_BUILD_MIXED_PREC: " ${JAMS_BUILD_MIXED_PREC})
message(STATUS "JAMS_BUILD_TESTS:      " ${JAMS_BUILD_TESTS})
message(STATUS "JAMS_BUILD_DOCS:       " ${JAMS_BUILD_MIXED_PREC})
message(STATUS "JAMS_BUILD_OFFLINE:    " ${JAMS_BUILD_OFFLINE})

message("compiler options:")
message(STATUS "CMAKE_BUILD_TYPE:         " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_CXX_COMPILER:       " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_LINKER:             " ${CMAKE_LINKER})
message(STATUS "CMAKE_MAKE_PROGRAM:       " ${CMAKE_MAKE_PROGRAM})
message(STATUS "CMAKE_CXX_COMPILE_FLAGS:        " ${CMAKE_CXX_COMPILE_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS:                " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS_DEBUG:          " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "CMAKE_CXX_FLAGS_RELEASE:        " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
message(STATUS "JAMS_FAST_MATH_OPT:             " ${JAMS_FAST_MATH_OPT})

if(JAMS_BUILD_CUDA)
message("cuda options:")
    message(STATUS "CUDA_VERSION:             " ${CUDA_VERSION})
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR:    " ${CUDA_TOOLKIT_ROOT_DIR})
    message(STATUS "CUDA_CUDART_LIBRARY:      " ${CUDA_CUDART_LIBRARY})
    message(STATUS "CUDA_HOST_COMPILER:       " ${CUDA_HOST_COMPILER})
    message(STATUS "CUDA_NVCC_EXECUTABLE:     " ${CUDA_NVCC_EXECUTABLE})
endif()



message("-----------------------------------------------------------")
