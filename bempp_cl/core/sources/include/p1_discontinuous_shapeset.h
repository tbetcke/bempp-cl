#ifndef bempp_p1_discontinuous_shapeset_h
#define bempp_p1_discontinuous_shapeset_h

#include "bempp_base_types.h"

inline void p1_discontinuous_evaluate(const REALTYPE2* localPoint, REALTYPE* result)
{
    result[0] = M_ONE - localPoint->x - localPoint->y;
    result[1] = localPoint->x;
    result[2] = localPoint->y;

}


#endif