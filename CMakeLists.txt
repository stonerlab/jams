# shameless extensive use of https://github.com/BVLC/caffe cmake files

cmake_minimum_required(VERSION 2.8.7)

#find_program(iwyu_path NAMES include-what-you-use iwyu)
#if(NOT iwyu_path)
#  message(FATAL_ERROR "Could not find the program include-what-you-use")
#endif()

# --- jams project
project(jams C CXX)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)


# -- Dependencies
include(cmake/Utils.cmake)
include(cmake/Dependencies.cmake)
include(cmake/Targets.cmake)


jams_ensure_out_of_source_build()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

#-----------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

#-----------------------------------------------------------

include_directories(include)

# TODO: make an xcode module to choose the include path automatically
include_directories(/Applications/Xcode-7.3.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include)

add_subdirectory(src/jams)

add_subdirectory(test/jams)