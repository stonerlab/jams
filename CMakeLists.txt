cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(${CMAKE_VERSION} VERSION_GREATER "3.12.0")
    cmake_policy(SET CMP0074 NEW) # ``find_package()`` uses ``<PackageName>_ROOT`` variables.
endif()

option(JAMS_BUILD_OMP "Build JAMS with OpenMP support" OFF)
option(JAMS_BUILD_CUDA "Build JAMS with CUDA support" ON)
option(JAMS_BUILD_FASTMATH "Build JAMS with fast math flags" ON)
option(JAMS_BUILD_TESTS "Build all of JAMS's unit tests." OFF)
option(JAMS_BUILD_MIXED_PREC "Build parts of JAMS using mixed precision." OFF)

option(USE_BOOST "Build HighFive with Boost support" OFF)
option(HIGHFIVE_EXAMPLES "Build HighFive examples" OFF)
option(HIGHFIVE_UNIT_TESTS "Build HighFive unit tests" OFF)

if (JAMS_BUILD_CUDA)
    project(jams LANGUAGES C CXX CUDA)
else()
    project(jams LANGUAGES C CXX)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

set(VERSION_MAJOR "2" )
set(VERSION_MINOR "0" )
set(VERSION_PATCH "0" )
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

include(cmake/Generate.cmake)
include(cmake/Utils.cmake)
include(cmake/Dependencies.cmake)

ensure_out_of_source_build()

add_subdirectory(src/jams)

if (JAMS_BUILD_TESTS)
    add_subdirectory(test/jams)
endif()