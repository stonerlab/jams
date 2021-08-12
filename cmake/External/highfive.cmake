include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

if (DEFINED JAMS_HIGHFIVE_VERSION)

    set(HIGHFIVE_USE_BOOST     OFF CACHE INTERNAL "Enable Boost Support")
    set(HIGHFIVE_USE_EIGEN     OFF CACHE INTERNAL "Enable Eigen testing")
    set(HIGHFIVE_USE_XTENSOR   OFF CACHE INTERNAL "Enable xtensor testing")
    set(HIGHFIVE_USE_OPENCV    OFF CACHE INTERNAL "Enable OpenCV testing")
    set(HIGHFIVE_UNIT_TESTS    OFF CACHE INTERNAL "Enable unit tests")
    set(HIGHFIVE_EXAMPLES      OFF CACHE INTERNAL "Compile examples")
    set(HIGHFIVE_PARALLEL_HDF5 OFF CACHE INTERNAL "Enable Parallel HDF5 support")
    set(HIGHFIVE_BUILD_DOCS    OFF CACHE INTERNAL "Enable documentation building")

    set(JAMS_HIGHFIVE_URL "https://github.com/BlueBrain/HighFive.git")
    if (MESSAGE_QUIET AND (NOT DEFINED VERBOSE))
        download_project(
                PROJ                HighFive
                GIT_REPOSITORY      ${JAMS_HIGHFIVE_URL}
                GIT_TAG             ${JAMS_HIGHFIVE_VERSION}
                GIT_SHALLOW         ON
                QUIET
        )
    else()
        download_project(
                PROJ                HighFive
                GIT_REPOSITORY      ${JAMS_HIGHFIVE_URL}
                GIT_TAG             ${JAMS_HIGHFIVE_VERSION}
                GIT_SHALLOW         ON
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