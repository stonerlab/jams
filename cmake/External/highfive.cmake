include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

if (DEFINED JAMS_HIGHFIVE_VERSION)
    set(USE_BOOST OFF CACHE INTERNAL "")
    set(HIGHFIVE_EXAMPLES OFF CACHE INTERNAL "")
    set(HIGHFIVE_UNIT_TESTS OFF CACHE INTERNAL "")

    set(PROJ_CMAKE_ARGS -DUSE_BOOST=OFF -DHIGHFIVE_EXAMPLES=OFF -DHIGHFIVE_UNIT_TESTS=OFF)

    set(JAMS_HIGHFIVE_URL "https://github.com/BlueBrain/HighFive.git")
    if (MESSAGE_QUIET AND (NOT DEFINED VERBOSE))
        download_project(
                PROJ                HighFive
                GIT_REPOSITORY      ${JAMS_HIGHFIVE_URL}
                GIT_TAG             ${JAMS_HIGHFIVE_VERSION}
                GIT_SHALLOW         ON
                CMAKE_ARGS          ${PROJ_CMAKE_ARGS}
                QUIET
        )
    else()
        download_project(
                PROJ                HighFive
                GIT_REPOSITORY      ${JAMS_HIGHFIVE_URL}
                GIT_TAG             ${JAMS_HIGHFIVE_VERSION}
                GIT_SHALLOW         ON
                CMAKE_ARGS          ${PROJ_CMAKE_ARGS}
        )
    endif()

    add_subdirectory(${HighFive_SOURCE_DIR} ${HighFive_BINARY_DIR} EXCLUDE_FROM_ALL)

    add_library(highfive ALIAS HighFive)
    set(JAMS_HIGHFIVE_LIBRARIES "built-in (git)")
else()
    find_package(HighFive REQUIRED)

    add_library(highfive INTERFACE)
    target_link_libraries(highfive INTERFACE HighFive)
    set(JAMS_HIGHFIVE_LIBRARIES ${HighFive_FOUND})
endif()