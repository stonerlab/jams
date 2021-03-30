include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

if (DEFINED JAMS_SPGLIB_VERSION)
    set(JAMS_SPGLIB_URL "https://github.com/spglib/spglib.git")
    if (MESSAGE_QUIET AND (NOT DEFINED VERBOSE))
        download_project(
                PROJ                spglib
                GIT_REPOSITORY      ${JAMS_SPGLIB_URL}
                GIT_TAG             ${JAMS_SPGLIB_VERSION}
                GIT_SHALLOW         ON
                QUIET
        )
    else()
        download_project(
                PROJ                spglib
                GIT_REPOSITORY      ${JAMS_SPGLIB_URL}
                GIT_TAG             ${JAMS_SPGLIB_VERSION}
                GIT_SHALLOW         ON
        )
    endif()

    add_subdirectory(${spglib_SOURCE_DIR} ${spglib_BINARY_DIR} EXCLUDE_FROM_ALL)
    target_include_directories(symspg PUBLIC $<BUILD_INTERFACE:${spglib_SOURCE_DIR}/src>)
    target_include_directories(symspg_static PUBLIC $<BUILD_INTERFACE:${spglib_SOURCE_DIR}/src>)

    add_library(spglib ALIAS symspg_static)
    set(JAMS_SPGLIB_LIBRARIES "built-in (git)")
else()
    find_package(SYMSPG REQUIRED)
    add_library(spglib INTERFACE)
    target_link_libraries(spglib INTERFACE ${SYMSPG_LIBRARY})
    target_include_directories(spglib INTERFACE ${SYMSPG_INCLUDE_DIR})
    set(JAMS_SPGLIB_LIBRARIES ${SYMSPG_LIBRARY})
endif()