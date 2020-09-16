include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)
include(CheckSymbolExists)

set(BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
set(BUILD_TESTS OFF CACHE INTERNAL "")

set(PROJ_CMAKE_ARGS -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF)

download_project(
        PROJ                libconfig
        GIT_REPOSITORY      https://github.com/hyperrealm/libconfig.git
        GIT_TAG             ${JAMS_LIBCONFIG_VERSION}
        CMAKE_ARGS          ${PROJ_CMAKE_ARGS}
)

add_subdirectory(${libconfig_SOURCE_DIR} ${libconfig_BINARY_DIR} EXCLUDE_FROM_ALL)
target_include_directories(config++ PUBLIC $<BUILD_INTERFACE:${libconfig_SOURCE_DIR}/lib>)

add_library(libconfig_builtin ALIAS config++)
set(JAMS_LIBCONFIG_LIBRARIES "built-in (git)")
