include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

download_project(
        PROJ                spglib
        GIT_REPOSITORY      https://github.com/atztogo/spglib.git
        GIT_TAG             ${JAMS_SPGLIB_VERSION}
        UPDATE_DISCONNECTED 1
)

add_subdirectory(${spglib_SOURCE_DIR} ${spglib_BINARY_DIR} EXCLUDE_FROM_ALL)
target_include_directories(symspg INTERFACE ${spglib_SOURCE_DIR}/src)
target_include_directories(symspg_static INTERFACE ${spglib_SOURCE_DIR}/src)

add_library(spglib_builtin ALIAS symspg_static)
set(JAMS_SPGLIB_LIBRARIES "built-in (git)")