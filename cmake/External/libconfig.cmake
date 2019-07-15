set(LIBCONFIG_VERSION 1.7.2)

if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

set(BUILD_EXAMPLES OFF)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_TESTS OFF)

include(${PROJECT_SOURCE_DIR}/cmake/DownloadProject/DownloadProject.cmake)
download_project(PROJ                libconfig
        GIT_REPOSITORY      https://github.com/hyperrealm/libconfig.git
        GIT_TAG             v${LIBCONFIG_VERSION}
        ${UPDATE_DISCONNECTED_IF_AVAILABLE}
        CMAKE_ARGS
            -DBUILD_EXAMPLES=FALSE
            -DBUILD_SHARED_LIBS=FALSE
            -DBUILD_TESTS=FALSE
        )

add_subdirectory(${libconfig_SOURCE_DIR} ${libconfig_BINARY_DIR} EXCLUDE_FROM_ALL)
