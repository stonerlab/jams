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

# -- hdf5
find_package(HDF5 COMPONENTS CXX REQUIRED)
add_library(hdf5 INTERFACE IMPORTED)
set_property(TARGET hdf5 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS} ${HDF5_CXX_INCLUDE_DIR})
set_property(TARGET hdf5 PROPERTY INTERFACE_LINK_LIBRARIES ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})

jams_ensure_out_of_source_build()

add_library(pcg INTERFACE)
target_include_directories(pcg INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include/pcg>
        )

include_directories(src)
include_directories(include)

add_subdirectory(src/jams)
add_subdirectory(test/jams)