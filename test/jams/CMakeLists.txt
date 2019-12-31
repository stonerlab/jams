enable_testing()

if (JAMS_BUILD_CUDA)
    # current versions of cmake mistakenly pass -pthread to nvcc at link time
    # but nvcc does not accept -pthread
    set(gtest_disable_pthreads ON)
endif()

include("${PROJECT_SOURCE_DIR}/cmake/External/gtest.cmake")

add_executable(jams-tests "jams_test.cc" $<TARGET_OBJECTS:jams-object-lib>)

target_compile_features(jams-tests PRIVATE cxx_std_11)

target_link_libraries(jams-tests gtest gmock fftw3 cblas pcg symspg_static config++ hdf5)

if(JAMS_BUILD_OMP AND OpenMP_CXX_FOUND)
    target_compile_definitions(jams-tests PRIVATE HAS_OMP=1)
    target_link_libraries(jams-tests OpenMP::OpenMP_CXX)
    target_compile_options(jams-tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
endif()

if(JAMS_BUILD_CUDA)
    if (JAMS_BUILD_MIXED_PREC)
        target_compile_definitions(jams-object-lib PRIVATE HAS_CUSPARSE_MIXED_PREC=1)
    endif()
    target_compile_definitions(jams-tests PRIVATE HAS_CUDA=1)
    target_link_libraries(jams-tests cusparse curand cublas cufft)
endif()

target_include_directories(jams-tests PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)
target_include_directories(jams-tests PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_include_directories(jams-tests PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/test>)

include(GoogleTest)

gtest_discover_tests(jams-tests)