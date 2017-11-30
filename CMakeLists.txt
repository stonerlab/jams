# shameless extensive use of https://github.com/BVLC/caffe cmake files
cmake_minimum_required(VERSION 2.8.7)
project(jams C CXX)
set(VERSION_MAJOR "2" )
set(VERSION_MINOR "0" )
set(VERSION_PATCH "0" )
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
include(cmake/Generate.cmake)
include(cmake/Utils.cmake)
include(cmake/Dependencies.cmake)
include(cmake/Targets.cmake)

jams_ensure_out_of_source_build()

include_directories(src)
include_directories(include)

add_subdirectory(src/jams)
add_subdirectory(test/jams)