#include "bempp_base_types.h"

__kernel void kernel_function(__global REALTYPE* inputBuffer,
                                     __global REALTYPE* resultBuffer,
                                     uint sumLength) {
    size_t gid = get_global_id(0);
    size_t index;

#ifndef COMPLEX_RESULT
    REALTYPE myResult = M_ZERO;

    for (index = 0; index < sumLength; ++index)
        myResult += inputBuffer[sumLength * gid + index];
    resultBuffer[gid] = myResult;
#else
    REALTYPE myResult[2];
    myResult[0] = M_ZERO;
    myResult[1] = M_ZERO;

    for (index = 0; index < sumLength; ++index) {
        myResult[0] += inputBuffer[2 * (sumLength * gid + index)];
        myResult[1] += inputBuffer[2 * (sumLength * gid + index) + 1];
    }
    resultBuffer[2 * gid] = myResult[0];
    resultBuffer[2 * gid + 1] = myResult[1];
#endif
}