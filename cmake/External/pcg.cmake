include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

set(JAMS_PCG_URL "https://github.com/imneme/pcg-cpp.git")
if (MESSAGE_QUIET AND (NOT DEFINED VERBOSE))
    download_project(
            PROJ                pcg
            GIT_REPOSITORY      ${JAMS_PCG_URL}
            GIT_TAG             ${JAMS_PCG_VERSION}
            QUIET
    )
else()
    download_project(
            PROJ                pcg
            GIT_REPOSITORY      ${JAMS_PCG_URL}
            GIT_TAG             ${JAMS_PCG_VERSION}
    )
endif()

add_library(pcg_builtin INTERFACE)
target_include_directories(pcg_builtin INTERFACE ${pcg_SOURCE_DIR}/include)

set(JAMS_PCG_LIBRARIES "built-in (git)")