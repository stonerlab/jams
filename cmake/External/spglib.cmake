set(SPGLIB_VERSION 1.10.4)

if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include(${PROJECT_SOURCE_DIR}/cmake/DownloadProject/DownloadProject.cmake)
download_project(PROJ                spglib
        GIT_REPOSITORY      https://github.com/atztogo/spglib.git
        GIT_TAG             v${SPGLIB_VERSION}
        ${UPDATE_DISCONNECTED_IF_AVAILABLE}
        )

add_subdirectory(${spglib_SOURCE_DIR} ${spglib_BINARY_DIR} EXCLUDE_FROM_ALL)

target_include_directories(symspg INTERFACE ${spglib_SOURCE_DIR}/src)
target_include_directories(symspg_static INTERFACE ${spglib_SOURCE_DIR}/src)
