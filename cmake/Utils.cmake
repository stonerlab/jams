macro(ensure_out_of_source_build)
     string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}"
     "${CMAKE_BINARY_DIR}" insource)
     get_filename_component(PARENTDIR ${CMAKE_SOURCE_DIR} PATH)
     string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}"
     "${PARENTDIR}" insourcesubdir)
    if(insource OR insourcesubdir)
        message(FATAL_ERROR "${CMAKE_PROJECT_NAME} requires an out of source build.")
    endif(insource OR insourcesubdir)
endmacro()

function(jams_extract_git_info)
    set(JAMS_GIT_BRANCH "unknown")
    set(JAMS_GIT_COMMIT_HASH "unknown")
    find_package(Git)

    if(GIT_FOUND)
        execute_process(
                COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE JAMS_GIT_BRANCH
                RESULT_VARIABLE __git_result)

        if(NOT ${__git_result} EQUAL 0)
            set(JAMS_GIT_BRANCH "unknown")
        endif()

        # Get the latest abbreviated commit hash of the working branch
        execute_process(
                COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE JAMS_GIT_COMMIT_HASH
                RESULT_VARIABLE __git_result)

        if(NOT ${__git_result} EQUAL 0)
            set(JAMS_GIT_COMMIT_HASH "unknown")
        endif()
    endif()

    set(JAMS_GIT_BRANCH ${JAMS_GIT_BRANCH} PARENT_SCOPE)
    set(JAMS_GIT_COMMIT_HASH ${JAMS_GIT_COMMIT_HASH} PARENT_SCOPE)
endfunction()

function(jams_set_fast_math target)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(FAST_MATH_OPT -Ofast -ffast-math)
        target_compile_options(${target} PUBLIC $<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:${FAST_MATH_OPT}>)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(FAST_MATH_OPT -fno-math-errno  -fno-rounding-math -fno-signaling-nans -fno-signed-zeros -fcx-limited-range)
        target_compile_options(${target} PUBLIC $<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:${FAST_MATH_OPT}>)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(FAST_MATH_OPT -ftz -msse3 -no-prec-div -fast -fp-model fast=2)
        target_compile_options(${target} PUBLIC $<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:${FAST_MATH_OPT}>)
    endif()
endfunction()

function(prepend variable prefix)
    set(__list "")
    foreach(__s ${${variable}})
        list(APPEND __list "${prefix}/${__s}")
    endforeach()
    set(${variable} "${__list}" PARENT_SCOPE)
endfunction()

# frmo caffe
function(jams_convert_absolute_paths variable)
    set(__dlist "")
    foreach(__s ${${variable}})
        get_filename_component(__abspath ${__s} ABSOLUTE)
        list(APPEND __list ${__abspath})
    endforeach()
    set(${variable} ${__list} PARENT_SCOPE)
endfunction()