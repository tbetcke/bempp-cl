#ifndef bempp_rwg0_shapeset_h
#define bempp_rwg0_shapeset_h

#include "bempp_base_types.h"

inline void rwg0_evaluate(const REALTYPE2* localPoint, REALTYPE* result)
{
    
    // Shape function on edge 0
    result[2 * 0 + 0] = localPoint->x;
    result[2 * 0 + 1] = localPoint->y - 1;

    // Shape function on edge 1
    result[2 * 1 + 0] = localPoint->x - 1;
    result[2 * 1 + 1] = localPoint->y;

    // Shape function on edge 2
    result[2 * 2 + 0] = localPoint->x;
    result[2 * 2 + 1] = localPoint->y; 

}


#endif