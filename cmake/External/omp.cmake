find_package(Threads QUIET)
find_package(OpenMP QUIET)

# For macos we have to pass the flags to cmake, this line separates them correctly
separate_arguments(OpenMP_CXX_FLAGS UNIX_COMMAND "${OpenMP_CXX_FLAGS}")

# For CMake < 3.9, we need to make the target ourselves
if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(Threads QUIET)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    target_compile_options(OpenMP::OpenMP_CXX INTERFACE ${OpenMP_CXX_FLAGS})
    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
    target_link_libraries(OpenMP::OpenMP_CXX INTERFACE ${OpenMP_CXX_FLAGS} Threads::Threads)
endif()