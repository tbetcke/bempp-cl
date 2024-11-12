#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPEVEC))) void
kernel_function(__global REALTYPE *grid,
                                __global uint* indices,
                                __global int* normalSigns,
                                __global REALTYPE *evalPoints,
                                __global REALTYPE *coefficients,
                                __constant REALTYPE* quadPoints,
                                __constant REALTYPE *quadWeights,
                                __global REALTYPE *globalResult,
				__global REALTYPE* kernel_parameters) {
  size_t gid[2];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);

#if VEC_LENGTH == 4
  size_t elementIndex[4] = {
      indices[4 * gid[1] + 0],
      indices[4 * gid[1] + 1],
      indices[4 * gid[1] + 2],
      indices[4 * gid[1] + 3]};
#elif VEC_LENGTH == 8
  size_t elementIndex[8] = {
      indices[8 * gid[1] + 0], 
      indices[8 * gid[1] + 1], 
      indices[8 * gid[1] + 2], 
      indices[8 * gid[1] + 3], 
      indices[8 * gid[1] + 4], 
      indices[8 * gid[1] + 5], 
      indices[8 * gid[1] + 6], 
      indices[8 * gid[1] + 7]};
#elif VEC_LENGTH == 16
  size_t elementIndex[16] = {
      indices[16 * gid[1] + 0],
      indices[16 * gid[1] + 1],
      indices[16 * gid[1] + 2],
      indices[16 * gid[1] + 3],
      indices[16 * gid[1] + 4],
      indices[16 * gid[1] + 5],
      indices[16 * gid[1] + 6],
      indices[16 * gid[1] + 7],
      indices[16 * gid[1] + 8],
      indices[16 * gid[1] + 9],
      indices[16 * gid[1] + 10],
      indices[16 * gid[1] + 11],
      indices[16 * gid[1] + 12],
      indices[16 * gid[1] + 13],
      indices[16 * gid[1] + 14],
      indices[16 * gid[1] + 15]};
#endif
          

  REALTYPE3 evalGlobalPoint;
  REALTYPEVEC surfaceGlobalPoint[3];

  REALTYPEVEC corners[3][3];
  REALTYPEVEC jacobian[2][3];
  REALTYPEVEC normal[3];
  REALTYPE3 dummy;

  REALTYPE2 point;

  REALTYPEVEC intElement;
  REALTYPE value[NUMBER_OF_SHAPE_FUNCTIONS];

  size_t quadIndex;
  size_t index;

  evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);

#ifndef COMPLEX_RESULT
  __local REALTYPEVEC localResult[WORKGROUP_SIZE];
  REALTYPEVEC myResult = M_ZERO;
#else
  __local REALTYPEVEC localResult[WORKGROUP_SIZE][2];
  REALTYPEVEC myResult[2] = {M_ZERO, M_ZERO};
#endif

#ifndef COMPLEX_KERNEL
  REALTYPEVEC kernelValue;
#else
  REALTYPEVEC kernelValue[2];
#endif

#ifndef COMPLEX_COEFFICIENTS
  REALTYPEVEC tempResult;
  REALTYPEVEC myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS];
  for (int index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index)
    myCoefficients[index] = (REALTYPEVEC)(
#if VEC_LENGTH == 4
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index]);
#elif VEC_LENGTH == 8
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + index]);
#elif VEC_LENGTH == 16
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[8] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[9] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[10] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[11] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[12] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[13] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[14] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[15] + index]);
#endif

#else
  REALTYPEVEC tempResult[2];
  REALTYPEVEC myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS][2];
  for (int index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index) {
#if VEC_LENGTH == 4
    myCoefficients[index][0] = (REALTYPEVEC)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index)]);
    myCoefficients[index][1] = (REALTYPEVEC)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index) +
                     1]);
#elif VEC_LENGTH == 8
    myCoefficients[index][0] = (REALTYPEVEC)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + index)]);
    myCoefficients[index][1] = (REALTYPEVEC)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + index) +
                     1]);
#elif VEC_LENGTH == 16
    myCoefficients[index][0] = (REALTYPE16)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[8] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[9] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[10] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[11] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[12] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[13] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[14] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[15] + index)]);
    myCoefficients[index][1] = (REALTYPE16)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[8] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[9] + index) +
                     1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[10] + index) + 1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[11] + index) + 1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[12] + index) + 1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[13] + index) + 1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[14] + index) + 1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[15] + index) + 1]);
#endif
  }
#endif

  getCornersVec(grid, elementIndex, corners);
  getJacobianVec(corners, jacobian);
  getNormalAndIntegrationElementVec(jacobian, normal, &intElement);
  updateNormalsVec(elementIndex, normalSigns, normal);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point = (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    BASIS(SHAPESET, evaluate)(&point, &value[0]);
    getGlobalPointVec(corners, &point, surfaceGlobalPoint);
#ifndef COMPLEX_KERNEL
    KERNEL(VEC_STRING)
    (evalGlobalPoint, surfaceGlobalPoint, dummy, normal, kernel_parameters, &kernelValue);
#else
    KERNEL(VEC_STRING)
    (evalGlobalPoint, surfaceGlobalPoint, dummy, normal, kernel_parameters, kernelValue);
#endif

#ifndef COMPLEX_COEFFICIENTS
    tempResult = M_ZERO;
    for (index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index)
      tempResult += myCoefficients[index] * value[index];
    tempResult *= quadWeights[quadIndex];
#ifndef COMPLEX_KERNEL
    myResult += tempResult * kernelValue;
#else
    myResult[0] += tempResult * kernelValue[0];
    myResult[1] += tempResult * kernelValue[1];
#endif
#else
    tempResult[0] = M_ZERO;
    tempResult[1] = M_ZERO;
    for (index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index) {
      tempResult[0] += myCoefficients[index][0] * value[index];
      tempResult[1] += myCoefficients[index][1] * value[index];
    }
    tempResult[0] *= quadWeights[quadIndex];
    tempResult[1] *= quadWeights[quadIndex];

#ifndef COMPLEX_KERNEL
    myResult[0] += tempResult[0] * kernelValue;
    myResult[1] += tempResult[1] * kernelValue;
#else
    myResult[0] +=
        tempResult[0] * kernelValue[0] - tempResult[1] * kernelValue[1];
    myResult[1] +=
        tempResult[0] * kernelValue[1] + tempResult[1] * kernelValue[0];
#endif
#endif
  }

#ifndef COMPLEX_RESULT
  localResult[lid] = myResult * intElement;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (index = 1; index < WORKGROUP_SIZE; ++index)
      localResult[0] += localResult[index];
    for (index = 0; index < VEC_LENGTH; ++index)
      globalResult[gid[0] * numGroups + groupId] +=
          ((__local REALTYPE *)(&localResult[0]))[index];
  }

#else
  localResult[lid][0] = myResult[0] * intElement;
  localResult[lid][1] = myResult[1] * intElement;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (index = 1; index < WORKGROUP_SIZE; ++index) {
      localResult[0][0] += localResult[index][0];
      localResult[0][1] += localResult[index][1];
    }
    for (index = 0; index < VEC_LENGTH; ++index) {
      globalResult[2 * (gid[0] * numGroups + groupId)] +=
          ((__local REALTYPE *)(&localResult[0][0]))[index];
      globalResult[2 * (gid[0] * numGroups + groupId) + 1] +=
          ((__local REALTYPE *)(&localResult[0][1]))[index];
    }
  }
#endif
}
