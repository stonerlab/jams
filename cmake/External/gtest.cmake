include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

set(JAMS_GTEST_URL "https://github.com/google/googletest.git")
if (MESSAGE_QUIET AND (NOT DEFINED VERBOSE))
    download_project(
            PROJ                googletest
            GIT_REPOSITORY      ${JAMS_GTEST_URL}
            GIT_TAG             ${JAMS_GTEST_VERSION}
            QUIET
    )
else()
    download_project(
            PROJ                googletest
            GIT_REPOSITORY      ${JAMS_GTEST_URL}
            GIT_TAG             ${JAMS_GTEST_VERSION}
    )
endif()


add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
