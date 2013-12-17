#ifndef SYS_ASSERT_H
#define SYS_ASSERT_H

#include <cassert>

#ifdef _DEBUG
#else
    #undef assert
    #define assert(x)   
#endif

#endif
