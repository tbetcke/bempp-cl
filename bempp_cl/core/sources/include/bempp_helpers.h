#ifndef bempp_helpers_h
#define bempp_helpers_h

#include "bempp_base_types.h"

inline void getCorners(__global REALTYPE *grid, size_t elementIndex, REALTYPE3 *corners)
{

    for (int i = 0; i < 3; ++i)
        corners[i] = (REALTYPE3)(grid[9 * elementIndex + 3 * i],
                                 grid[9 * elementIndex + 3 * i + 1],
                                 grid[9 * elementIndex + 3 * i + 2]);
}

inline void getLocalMultipliers(__global REALTYPE *localMultipliers, size_t elementIndex, REALTYPE *myMultipliers, int count)
{
    for (int i = 0; i < count; ++i)
        myMultipliers[i] = localMultipliers[count * elementIndex + i];
}

inline void getElement(__global uint *connectivity, size_t elementIndex, uint *element)
{
    for (int i = 0; i < 3; ++i)
        element[i] = connectivity[3 * elementIndex + i];
}

inline void getLocal2Global(__global uint *local2Global, size_t elementIndex, uint *myLocal2Global, int count)
{
    for (int i = 0; i < count; ++i)
        myLocal2Global[i] = local2Global[count * elementIndex + i];
}

inline void getJacobian(REALTYPE3 *corners, REALTYPE3 *jacobian)
{
    jacobian[0] = corners[1] - corners[0];
    jacobian[1] = corners[2] - corners[0];
}

inline void getNormalAndIntegrationElement(REALTYPE3 *jacobian, REALTYPE3 *normal, REALTYPE *integrationElement)
{
    *normal = cross(jacobian[0], jacobian[1]);
    *integrationElement = length(*normal);
    *normal /= *integrationElement;
}

inline REALTYPE3 getGlobalPoint(REALTYPE3 *corners, REALTYPE2 *localPoint)
{
    return corners[0] * (M_ONE - localPoint->x - localPoint->y) + corners[1] * localPoint->x + corners[2] * localPoint->y;
}

inline void getPiolaTransform(REALTYPE integrationElement, REALTYPE3 jacobian[2],
                              REALTYPE referenceValues[3][2], REALTYPE3 result[3])
{
    for (int i = 0; i < 3; ++i)
        result[i] = M_ONE / integrationElement * (jacobian[0] * referenceValues[i][0] + jacobian[1] * referenceValues[i][1]);
}

inline void computeEdgeLength(REALTYPE3 corners[3], REALTYPE result[3])
{

    result[0] = distance(corners[1], corners[0]);
    result[1] = distance(corners[2], corners[0]);
    result[2] = distance(corners[2], corners[1]);
}

inline void updateNormals(size_t index, __global int *signs, REALTYPE3 *normal)
{

    *normal *= signs[index];
}

#ifdef REALTYPEVEC

inline void getElementVec(__global uint *connectivity, size_t *elementIndex, uint element[VEC_LENGTH][3])
{
    for (int i = 0; i < VEC_LENGTH; ++i)
        for (int j = 0; j < 3; ++j)
            element[i][j] = connectivity[3 * elementIndex[i] + j];
}

inline void getLocal2GlobalVec(__global uint *local2Global, size_t *elementIndex, uint *myLocal2Global, int count)
{
    for (int i = 0; i < VEC_LENGTH; ++i)
        for (int j = 0; j < count; ++j)
            myLocal2Global[count * i + j] = local2Global[count * elementIndex[i] + j];
}

inline void getLocalMultipliersVec(__global REALTYPE *localMultipliers, size_t *elementIndex, REALTYPE *myMultipliers, int count)
{
    for (int i = 0; i < VEC_LENGTH; ++i)
        for (int j = 0; j < count; ++j)
            myMultipliers[count * i + j] = localMultipliers[count * elementIndex[i] + j];
}

inline void getNormalAndIntegrationElementVec(REALTYPEVEC jacobian[2][3], REALTYPEVEC normal[3], REALTYPEVEC *integrationElement)
{

    normal[0] = jacobian[0][1] * jacobian[1][2] - jacobian[0][2] * jacobian[1][1];
    normal[1] = jacobian[0][2] * jacobian[1][0] - jacobian[0][0] * jacobian[1][2];
    normal[2] = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];

    *integrationElement = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    normal[0] /= *integrationElement;
    normal[1] /= *integrationElement;
    normal[2] /= *integrationElement;
}

inline void getGlobalPointVec(REALTYPEVEC corners[3][3], REALTYPE2 *localPoint, REALTYPEVEC globalPoint[3])
{
    globalPoint[0] = corners[0][0] * (M_ONE - localPoint->x - localPoint->y) + corners[1][0] * localPoint->x + corners[2][0] * localPoint->y;
    globalPoint[1] = corners[0][1] * (M_ONE - localPoint->x - localPoint->y) + corners[1][1] * localPoint->x + corners[2][1] * localPoint->y;
    globalPoint[2] = corners[0][2] * (M_ONE - localPoint->x - localPoint->y) + corners[1][2] * localPoint->x + corners[2][2] * localPoint->y;
}

inline void getPiolaTransformVec(REALTYPEVEC integrationElement, REALTYPEVEC jacobian[2][3],
                                 REALTYPE referenceValues[3][2], REALTYPEVEC result[3][3])
{
    for (int base = 0; base < 3; ++base)
        for (int row = 0; row < 3; ++row)
            result[base][row] = M_ONE / integrationElement * (jacobian[0][row] * referenceValues[base][0] + jacobian[1][row] * referenceValues[base][1]);
}

inline void computeEdgeLengthVec(REALTYPEVEC corners[3][3], REALTYPEVEC result[3])
{

    REALTYPEVEC diff;
    int j;

    result[0] = M_ZERO;
    result[1] = M_ZERO;
    result[2] = M_ZERO;

    for (j = 0; j < 3; ++j)
    {
        diff = corners[1][j] - corners[0][j];
        result[0] += diff * diff;
    }

    for (j = 0; j < 3; ++j)
    {
        diff = corners[2][j] - corners[0][j];
        result[1] += diff * diff;
    }

    for (j = 0; j < 3; ++j)
    {
        diff = corners[2][j] - corners[1][j];
        result[2] += diff * diff;
    }

    result[0] = sqrt(result[0]);
    result[1] = sqrt(result[1]);
    result[2] = sqrt(result[2]);
}

#if VEC_LENGTH == 4
inline void getCornersVec(__global REALTYPE *grid, size_t *elementIndex, REALTYPE4 corners[3][3])
{
    /* corners[i][j] is the jth element of the ith corner in each of the 4 elements
       in the elementIndex array */

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            corners[i][j] = (REALTYPE4)(grid[9 * elementIndex[0] + 3 * i + j],
                                        grid[9 * elementIndex[1] + 3 * i + j],
                                        grid[9 * elementIndex[2] + 3 * i + j],
                                        grid[9 * elementIndex[3] + 3 * i + j]);
}

inline void getJacobianVec(REALTYPE4 corners[3][3], REALTYPE4 jacobian[2][3])
{
    jacobian[0][0] = corners[1][0] - corners[0][0];
    jacobian[0][1] = corners[1][1] - corners[0][1];
    jacobian[0][2] = corners[1][2] - corners[0][2];

    jacobian[1][0] = corners[2][0] - corners[0][0];
    jacobian[1][1] = corners[2][1] - corners[0][1];
    jacobian[1][2] = corners[2][2] - corners[0][2];
}

inline void updateNormalsVec(size_t index[VEC_LENGTH], __global int *signs, REALTYPEVEC normal[3])
{

    REALTYPEVEC signFlip = (REALTYPEVEC)(signs[index[0]], signs[index[1]],
                                         signs[index[2]], signs[index[3]]);

    normal[0] *= signFlip;
    normal[1] *= signFlip;
    normal[2] *= signFlip;
}

#elif VEC_LENGTH == 8
inline void getCornersVec(__global REALTYPE *grid, size_t *elementIndex, REALTYPE8 corners[3][3])
{
    /* corners[i][j] is the jth element of the ith corner in each of the 4 elements
       in the elementIndex array */

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            corners[i][j] = (REALTYPE8)(grid[9 * elementIndex[0] + 3 * i + j],
                                        grid[9 * elementIndex[1] + 3 * i + j],
                                        grid[9 * elementIndex[2] + 3 * i + j],
                                        grid[9 * elementIndex[3] + 3 * i + j],
                                        grid[9 * elementIndex[4] + 3 * i + j],
                                        grid[9 * elementIndex[5] + 3 * i + j],
                                        grid[9 * elementIndex[6] + 3 * i + j],
                                        grid[9 * elementIndex[7] + 3 * i + j]);
}

inline void getJacobianVec(REALTYPE8 corners[3][3], REALTYPE8 jacobian[2][3])
{
    jacobian[0][0] = corners[1][0] - corners[0][0];
    jacobian[0][1] = corners[1][1] - corners[0][1];
    jacobian[0][2] = corners[1][2] - corners[0][2];

    jacobian[1][0] = corners[2][0] - corners[0][0];
    jacobian[1][1] = corners[2][1] - corners[0][1];
    jacobian[1][2] = corners[2][2] - corners[0][2];
}

inline void updateNormalsVec(size_t index[VEC_LENGTH], __global int *signs, REALTYPEVEC normal[3])
{

    REALTYPEVEC signFlip = (REALTYPEVEC)(signs[index[0]], signs[index[1]],
                                         signs[index[2]], signs[index[3]],
                                         signs[index[4]], signs[index[5]],
                                         signs[index[6]], signs[index[7]]);

    normal[0] *= signFlip;
    normal[1] *= signFlip;
    normal[2] *= signFlip;
}

#elif VEC_LENGTH == 16
inline void getCornersVec(__global REALTYPE *grid, size_t *elementIndex, REALTYPE16 corners[3][3])
{
    /* corners[i][j] is the jth element of the ith corner in each of the 4 elements
       in the elementIndex array */

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            corners[i][j] = (REALTYPE16)(grid[9 * elementIndex[0] + 3 * i + j],
                                         grid[9 * elementIndex[1] + 3 * i + j],
                                         grid[9 * elementIndex[2] + 3 * i + j],
                                         grid[9 * elementIndex[3] + 3 * i + j],
                                         grid[9 * elementIndex[4] + 3 * i + j],
                                         grid[9 * elementIndex[5] + 3 * i + j],
                                         grid[9 * elementIndex[6] + 3 * i + j],
                                         grid[9 * elementIndex[7] + 3 * i + j],
                                         grid[9 * elementIndex[8] + 3 * i + j],
                                         grid[9 * elementIndex[9] + 3 * i + j],
                                         grid[9 * elementIndex[10] + 3 * i + j],
                                         grid[9 * elementIndex[11] + 3 * i + j],
                                         grid[9 * elementIndex[12] + 3 * i + j],
                                         grid[9 * elementIndex[13] + 3 * i + j],
                                         grid[9 * elementIndex[14] + 3 * i + j],
                                         grid[9 * elementIndex[15] + 3 * i + j]);
}

inline void getJacobianVec(REALTYPE16 corners[3][3], REALTYPE16 jacobian[2][3])
{
    jacobian[0][0] = corners[1][0] - corners[0][0];
    jacobian[0][1] = corners[1][1] - corners[0][1];
    jacobian[0][2] = corners[1][2] - corners[0][2];

    jacobian[1][0] = corners[2][0] - corners[0][0];
    jacobian[1][1] = corners[2][1] - corners[0][1];
    jacobian[1][2] = corners[2][2] - corners[0][2];
}

inline void updateNormalsVec(size_t index[VEC_LENGTH], __global int *signs, REALTYPEVEC normal[3])
{

    REALTYPEVEC signFlip = (REALTYPEVEC)(signs[index[0]], signs[index[1]],
                                         signs[index[2]], signs[index[3]],
                                         signs[index[4]], signs[index[5]],
                                         signs[index[6]], signs[index[7]],
                                         signs[index[8]], signs[index[9]],
                                         signs[index[10]], signs[index[11]],
                                         signs[index[12]], signs[index[13]],
                                         signs[index[14]], signs[index[15]]);

    normal[0] *= signFlip;
    normal[1] *= signFlip;
    normal[2] *= signFlip;
}

#endif

#endif

inline uint elementsAreAdjacent(uint *element1, uint *element2, bool gridsAreDisjoint)
{
    return !gridsAreDisjoint &&
           (element1[0] == element2[0] || element1[0] == element2[1] || element1[0] == element2[2] ||
            element1[1] == element2[0] || element1[1] == element2[1] || element1[1] == element2[2] ||
            element1[2] == element2[0] || element1[2] == element2[1] || element1[2] == element2[2]);
}

inline uint elementsAreAdjacentCollocation(uint *element1, uint *element2, bool gridsAreDisjoint)
{
    return !gridsAreDisjoint &&
           (element1[0] == element2[0] && element1[1] == element2[1] && element1[2] == element2[2]);
}

#endif
