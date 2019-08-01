# https://github.com/Crascit/DownloadProject/blob/master/CMakeLists.txt
if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include(${PROJECT_SOURCE_DIR}/cmake/DownloadProject/DownloadProject.cmake)
download_project(PROJ                pcg
        GIT_REPOSITORY      https://github.com/imneme/pcg-cpp.git
        GIT_TAG             v0.98.1
        ${UPDATE_DISCONNECTED_IF_AVAILABLE}
        )

add_library(pcg INTERFACE IMPORTED)
set_property(TARGET pcg PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${pcg_SOURCE_DIR}/include)
