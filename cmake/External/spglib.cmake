include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

if (DEFINED JAMS_SPGLIB_VERSION)

    set(USE_OMP ${JAMS_BUILD_OMP} CACHE INTERNAL "Build with OMP support")

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

    if (DEFINED CMAKE_POLICY_VERSION_MINIMUM)
        set(_jams_policy_minimum_backup "${CMAKE_POLICY_VERSION_MINIMUM}")
    endif()
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    add_subdirectory(${spglib_SOURCE_DIR} ${spglib_BINARY_DIR} EXCLUDE_FROM_ALL)
    if (DEFINED _jams_policy_minimum_backup)
        set(CMAKE_POLICY_VERSION_MINIMUM "${_jams_policy_minimum_backup}")
        unset(_jams_policy_minimum_backup)
    else()
        unset(CMAKE_POLICY_VERSION_MINIMUM)
    endif()
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
