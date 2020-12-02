cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(${CMAKE_VERSION} VERSION_GREATER "3.12.0")
    cmake_policy(SET CMP0074 NEW) # ``find_package()`` uses ``<PackageName>_ROOT`` variables.
endif()

include(cmake/Utils.cmake)
ensure_out_of_source_build()

option(JAMS_BUILD_OFFLINE "Build JAMS without updating external dependencies online" OFF)
option(JAMS_BUILD_OMP "Build JAMS with OpenMP support" OFF)
option(JAMS_BUILD_CUDA "Build JAMS with CUDA support" ON)
option(JAMS_BUILD_FASTMATH "Build JAMS with fast math flags" ON)
option(JAMS_BUILD_MIXED_PREC "Build parts of JAMS using mixed precision." OFF)
option(JAMS_BUILD_TESTS "Build all of JAMS's unit tests." OFF)
option(JAMS_BUILD_DOCS "Build all of JAMS's documentation." OFF)

project(jams LANGUAGES C CXX)

if (JAMS_BUILD_CUDA)
    enable_language(CUDA)
endif()

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
set(VERSION_MINOR "7" )
set(VERSION_PATCH "0" )
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# Latest libconfig version is 1.7.2 but the cmake support is broken. This is fixed in the current
# github master branch
set(JAMS_LIBCONFIG_VERSION f53e5de) # https://github.com/hyperrealm/libconfig
set(JAMS_HIGHFIVE_VERSION v2.1.1)   # https://github.com/BlueBrain/HighFive
set(JAMS_SPGLIB_VERSION v1.16.0)    # https://github.com/atztogo/spglib
set(JAMS_PCG_VERSION v0.98.1)       # https://github.com/imneme/pcg-cpp

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
