add_library(pcg INTERFACE)

if (DEFINED JAMS_PCG_VERSION)

    include(${PROJECT_SOURCE_DIR}/cmake/Modules/DownloadProject.cmake)

    set(JAMS_PCG_URL "https://github.com/imneme/pcg-cpp.git")
    if (MESSAGE_QUIET AND (NOT DEFINED VERBOSE))
        download_project(
                PROJ                pcg
                GIT_REPOSITORY      ${JAMS_PCG_URL}
                GIT_TAG             ${JAMS_PCG_VERSION}
                GIT_SHALLOW         ON
                QUIET
        )
    else()
        download_project(
                PROJ                pcg
                GIT_REPOSITORY      ${JAMS_PCG_URL}
                GIT_TAG             ${JAMS_PCG_VERSION}
                GIT_SHALLOW         ON
        )
    endif()

    target_include_directories(pcg INTERFACE ${pcg_SOURCE_DIR}/include)

    set(JAMS_PCG_LIBRARIES "built-in (git)")
else()
    find_path(JAMS_PCG_LIBRARIES pcg_random.hpp PATHS ENV CPATH /usr/include /usr/local/include)
    if (NOT JAMS_PCG_LIBRARIES)
        message(FATAL_ERROR "Could not find pcg header files")
    endif()
    target_include_directories(pcg INTERFACE ${JAMS_PCG_LIBRARIES})
endif()

