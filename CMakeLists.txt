# shameless extensive use of https://github.com/BVLC/caffe cmake files

cmake_minimum_required(VERSION 2.8.7)

# --- jams project
project(jams C CXX)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)


# -- Dependencies
include(cmake/Utils.cmake)
include(cmake/Dependencies.cmake)
include(cmake/Targets.cmake)

jams_ensure_out_of_source_build()


#-----------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

#-----------------------------------------------------------

include_directories(include)

add_subdirectory(src/jams)