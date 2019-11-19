include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

download_project(PROJ                googletest
                 GIT_REPOSITORY      https://github.com/google/googletest.git
                 GIT_TAG             master
                 UPDATE_DISCONNECTED 1)

add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
