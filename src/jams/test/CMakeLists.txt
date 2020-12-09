enable_testing()
include(../../../cmake/Sources.cmake)
include(GoogleTest)

if (JAMS_BUILD_CUDA)
    # current versions of cmake mistakenly pass -pthread to nvcc at link time
    # but nvcc does not accept -pthread
    set(gtest_disable_pthreads ON)
endif()

add_executable(jams_tests "jams_test.cc")

prepend(JAMS_SOURCES_CXX ..)
prepend(JAMS_SOURCES_CUDA ..)

target_sources(jams_tests PRIVATE ${JAMS_SOURCES_CXX})

if(JAMS_BUILD_CUDA)
    target_sources(jams_tests PRIVATE ${JAMS_SOURCES_CUDA})
endif()

target_link_libraries(jams_tests
        gtest
        gmock
        fftw_external
        cblas_external
        pcg_builtin
        spglib_builtin
        libconfig_builtin
        highfive_builtin)

target_compile_features(jams_tests PRIVATE cxx_std_11)

target_include_directories(jams_tests PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)

if (MKL_FOUND)
    target_compile_definitions(jams_tests PRIVATE HAS_MKL=1)
endif()

if (JAMS_BUILD_OMP AND OpenMP_CXX_FOUND)
    target_compile_definitions(jams_tests PRIVATE HAS_OMP=1)
    target_compile_options(jams_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    target_link_libraries(jams_tests OpenMP::OpenMP_CXX)
endif()

if(JAMS_BUILD_CUDA)
    target_compile_definitions(jams_tests PRIVATE HAS_CUDA=1)
    target_link_libraries(jams_tests
            cuda
            cusparse
            curand
            cublas
            cufft)

    if (JAMS_BUILD_MIXED_PREC)
        target_compile_definitions(jams_tests PRIVATE HAS_CUSPARSE_MIXED_PREC=1)
    endif()

    if(APPLE)
        # We need to add the path to the driver (libcuda.dylib) as an rpath,
        # so that the static cuda runtime can find it at runtime.
        set_target_properties(jams_tests PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
endif()

if (JAMS_BUILD_FASTMATH)
    jams_set_fast_math(jams_tests)
endif()

target_include_directories(jams_tests PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)
target_include_directories(jams_tests PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_include_directories(jams_tests PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/test>)


gtest_discover_tests(jams_tests)