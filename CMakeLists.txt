cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

option(jams_build_tests "Build all of JAMS's unit tests." OFF)

project(jams LANGUAGES C CXX CUDA)
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

if (jams_build_tests)
    add_subdirectory(test/jams)
endif()