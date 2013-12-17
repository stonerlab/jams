#ifndef JB_ARRAY_H
#define JB_ARRAY_H

#include "../sys/sys_assert.h"
#include "../sys/sys_defines.h"
#include "../sys/sys_types.h"
#include "../sys/sys_intrinsics.h"

#include <cstring>
#include <utility>
#include <algorithm>

template <typename type, uint32 dimension, typename index=uint32>
class Array 
{
    public:
        Array(){}
        ~Array(){};
};

#include "Array1D.h"
#include "Array2D.h"
#include "Array3D.h"

#endif
