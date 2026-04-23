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
        hdf5_external
        pcg
        spglib
        libconfig
        highfive)

target_compile_features(jams_tests PRIVATE cxx_std_14)

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
            cuda_external
            cusparse_external
            curand_external
            cublas_external
            cufft_external)

    if (JAMS_BUILD_MIXED_PREC)
        target_compile_definitions(jams_tests PRIVATE HAS_CUSPARSE_MIXED_PREC=1)
    endif()

    if(APPLE)
        # We need to add the path to the driver (libcuda.dylib) as an rpath,
        # so that the static cuda runtime can find it at runtime.
        set_target_properties(jams_tests PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
endif()


if (JAMS_BUILD_MIXED_PREC)
    target_compile_definitions(jams_tests PRIVATE DO_MIXED_PRECISION=1)
endif()

if (JAMS_BUILD_FASTMATH)
    jams_set_fast_math(jams_tests)
endif()

target_include_directories(jams_tests PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)
target_include_directories(jams_tests PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_include_directories(jams_tests PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/test>)

set(jams_discovered_tests)
gtest_discover_tests(
        jams_tests
        TEST_LIST jams_discovered_tests)

set(jams_test_labels_script "${CMAKE_CURRENT_BINARY_DIR}/jams_test_labels.cmake")
file(WRITE "${jams_test_labels_script}" [=[
foreach(test_name IN LISTS jams_discovered_tests)
    set(test_labels unit)
    set(run_serial FALSE)

    if (test_name MATCHES "GPUTests"
            OR test_name MATCHES "(^|[./])Gpu"
            OR test_name MATCHES "(^|[./])Cuda"
            OR test_name MATCHES "_gpu")
        list(APPEND test_labels gpu)
        set(run_serial TRUE)
    else()
        list(APPEND test_labels cpu)
    endif()

    if (test_name MATCHES "SLOW")
        list(APPEND test_labels slow)
    endif()

    set_tests_properties("${test_name}" PROPERTIES LABELS "${test_labels}")

    if (run_serial)
        set_tests_properties("${test_name}" PROPERTIES RUN_SERIAL TRUE)
    endif()
endforeach()
]=])
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${jams_test_labels_script}")
