include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

set(JAMS_SPGLIB_URL "https://github.com/spglib/spglib.git")
if (MESSAGE_QUIET AND (NOT DEFINED VERBOSE))
    download_project(
            PROJ                spglib
            GIT_REPOSITORY      ${JAMS_SPGLIB_URL}
            GIT_TAG             ${JAMS_SPGLIB_VERSION}
            QUIET
    )
else()
    download_project(
            PROJ                spglib
            GIT_REPOSITORY      ${JAMS_SPGLIB_URL}
            GIT_TAG             ${JAMS_SPGLIB_VERSION}
    )
endif()

add_subdirectory(${spglib_SOURCE_DIR} ${spglib_BINARY_DIR} EXCLUDE_FROM_ALL)
target_include_directories(symspg PUBLIC $<BUILD_INTERFACE:${spglib_SOURCE_DIR}/src>)
target_include_directories(symspg_static PUBLIC $<BUILD_INTERFACE:${spglib_SOURCE_DIR}/src>)

add_library(spglib_builtin ALIAS symspg_static)
set(JAMS_SPGLIB_LIBRARIES "built-in (git)")