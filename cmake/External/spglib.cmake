# https://github.com/Crascit/DownloadProject/blob/master/CMakeLists.txt
if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include(${PROJECT_SOURCE_DIR}/cmake/DownloadProject/DownloadProject.cmake)
download_project(PROJ                spglib
        GIT_REPOSITORY      https://github.com/atztogo/spglib.git
        GIT_TAG             v1.12.2
        ${UPDATE_DISCONNECTED_IF_AVAILABLE}
        )

add_subdirectory(${spglib_SOURCE_DIR} ${spglib_BINARY_DIR} EXCLUDE_FROM_ALL)