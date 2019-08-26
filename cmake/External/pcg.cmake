include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

download_project(
        PROJ                pcg
        GIT_REPOSITORY      https://github.com/imneme/pcg-cpp.git
        GIT_TAG             ${JAMS_PCG_VERSION}
        UPDATE_DISCONNECTED 1
)

add_library(pcg_builtin INTERFACE)
target_include_directories(pcg_builtin INTERFACE ${pcg_SOURCE_DIR}/include)

set(JAMS_PCG_LIBRARIES "built-in (git)")