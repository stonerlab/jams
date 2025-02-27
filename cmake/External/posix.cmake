include(CheckIncludeFiles)
check_include_files(sys/stat.h HAVE_SYS_STAT_HEADER)
if(NOT HAVE_SYS_STAT_HEADER)
    message(FATAL_ERROR "Cannot find sys/stat.h posix header")
endif()

check_include_files(unistd.h HAVE_UNISTD_HEADERS)
if(NOT HAVE_UNISTD_HEADERS)
    message(FATAL_ERROR "Cannot find required unistd posix header")
endif()

