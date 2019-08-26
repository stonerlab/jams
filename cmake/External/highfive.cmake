include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

set(USE_BOOST OFF CACHE INTERNAL "")
set(HIGHFIVE_EXAMPLES OFF CACHE INTERNAL "")
set(HIGHFIVE_UNIT_TESTS OFF CACHE INTERNAL "")

set(PROJ_CMAKE_ARGS -DUSE_BOOST=OFF -DHIGHFIVE_EXAMPLES=OFF -DHIGHFIVE_UNIT_TESTS=OFF)

download_project(
        PROJ                HighFive
        GIT_REPOSITORY      https://github.com/BlueBrain/HighFive.git
        GIT_TAG             ${JAMS_HIGHFIVE_VERSION}
        UPDATE_DISCONNECTED 1
        CMAKE_ARGS          ${PROJ_CMAKE_ARGS}
)

add_subdirectory(${HighFive_SOURCE_DIR} ${HighFive_BINARY_DIR} EXCLUDE_FROM_ALL)

add_library(highfive_builtin ALIAS HighFive)
set(JAMS_HIGHFIVE_LIBRARIES "built-in (git)")
