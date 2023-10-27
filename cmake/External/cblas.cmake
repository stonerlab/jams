add_library(cblas_external INTERFACE)

find_package(MKL QUIET)
if(MKL_FOUND)
    target_include_directories(cblas_external INTERFACE ${MKL_INCLUDE_DIR})
    target_link_libraries(cblas_external INTERFACE ${MKL_LIBRARIES})
    set(JAMS_CBLAS_FOUND true)
    set(JAMS_CBLAS_VENDOR "mkl")
    set(BLAS_FOUND ON)
endif()

if (NOT BLAS_FOUND)
    set(BLA_VENDOR Apple)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)

        file(TO_CMAKE_PATH "$ENV{Accelerate_HOME}" Accelerate_HOME)
        SET(Accelerate_INCLUDE_SEARCH_PATHS
                /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/Current
                /System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current
                ${Accelerate_HOME}
        )

        find_path(CBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Accelerate_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES Headers)

        target_include_directories(cblas_external INTERFACE ${CBLAS_INCLUDE_DIR})
        target_link_libraries(cblas_external INTERFACE ${BLAS_LIBRARIES})
        set(JAMS_CBLAS_FOUND true)
        set(JAMS_CBLAS_VENDOR "Apple")
    endif()
endif()

if (NOT BLAS_FOUND)
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        file(TO_CMAKE_PATH "$ENV{OpenBLAS_HOME}" OpenBLAS_HOME)

        SET(Open_BLAS_INCLUDE_SEARCH_PATHS
                $ENV{BLAS_ROOT}
                $ENV{BLAS_ROOT}/include
                $ENV{BLAS_ROOT}/include/openblas
                ${OpenBLAS_HOME}
                ${OpenBLAS_HOME}/include
                ${OpenBLAS_HOME}/include/openblas
                /usr/local/include/openblas
                /usr/local/include/openblas-base
                /usr/local/opt/openblas/include
                /opt/OpenBLAS/include
                /usr/include/openblas
                /usr/include/openblas-base
                ${PROJECT_SOURCE_DIR}/3rdparty/OpenBLAS/include
                ${PROJECT_SOURCE_DIR}/thirdparty/OpenBLAS/include
                /usr/include
                /usr/local/include
        )
        find_path(CBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES openblas)

        target_include_directories(cblas_external INTERFACE ${CBLAS_INCLUDE_DIR})
        target_link_libraries(cblas_external INTERFACE ${BLAS_LIBRARIES})
        set(JAMS_CBLAS_FOUND true)
        set(JAMS_CBLAS_VENDOR "OpenBLAS")
    endif()
endif()

if (NOT BLAS_FOUND)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        file(TO_CMAKE_PATH "$ENV{BLAS_HOME}" BLAS_HOME)
        SET(BLAS_INCLUDE_SEARCH_PATHS
                ${BLAS_HOME}
                ${BLAS_HOME}/include
                /usr/include
                /usr/local/include
        )
        find_path(CBLAS_INCLUDE_DIR NAMES cblas.h HINTS ${BLAS_INCLUDE_SEARCH_PATHS})

        target_include_directories(cblas_external INTERFACE ${CBLAS_INCLUDE_DIR})
        target_link_libraries(cblas_external INTERFACE ${BLAS_LIBRARIES})
        set(JAMS_CBLAS_FOUND true)
        set(JAMS_CBLAS_VENDOR "generic")
    endif()
endif()

if (NOT BLAS_FOUND)
    message(FATAL_ERROR "Failed to find a BLAS library")
endif()
