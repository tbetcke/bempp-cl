#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPEVEC))) void kernel_function(
    __global REALTYPE *grid, __global uint *indices, __global int *normalSigns,
    __global REALTYPE *evalPoints, __global REALTYPE *coefficients,
    __constant REALTYPE *quadPoints, __constant REALTYPE *quadWeights,
    __global REALTYPE *globalResult, __global REALTYPE *kernel_parameters) {
  size_t gid[2];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);

#if VEC_LENGTH == 4
  size_t elementIndex[4] = {indices[4 * gid[1] + 0], indices[4 * gid[1] + 1],
                            indices[4 * gid[1] + 2], indices[4 * gid[1] + 3]};
#elif VEC_LENGTH == 8
  size_t elementIndex[8] = {indices[8 * gid[1] + 0], indices[8 * gid[1] + 1],
                            indices[8 * gid[1] + 2], indices[8 * gid[1] + 3],
                            indices[8 * gid[1] + 4], indices[8 * gid[1] + 5],
                            indices[8 * gid[1] + 6], indices[8 * gid[1] + 7]};
#elif VEC_LENGTH == 16
  size_t elementIndex[16] = {
      indices[16 * gid[1] + 0],  indices[16 * gid[1] + 1],
      indices[16 * gid[1] + 2],  indices[16 * gid[1] + 3],
      indices[16 * gid[1] + 4],  indices[16 * gid[1] + 5],
      indices[16 * gid[1] + 6],  indices[16 * gid[1] + 7],
      indices[16 * gid[1] + 8],  indices[16 * gid[1] + 9],
      indices[16 * gid[1] + 10], indices[16 * gid[1] + 11],
      indices[16 * gid[1] + 12], indices[16 * gid[1] + 13],
      indices[16 * gid[1] + 14], indices[16 * gid[1] + 15]};
#endif

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);

  size_t vecIndex;

  REALTYPEVEC surfaceGlobalPoint[3];

  REALTYPE basisValue[3][2];
  REALTYPEVEC elementValue[3][3];

  REALTYPEVEC corners[3][3];
  REALTYPEVEC jacobian[2][3];
  REALTYPEVEC normal[3];

  REALTYPEVEC factor1[2];
  REALTYPE3 testNormal; // Dummy variable. Only needed for kernel call.

  REALTYPE2 point;

  REALTYPEVEC intElem;

  size_t quadIndex;
  size_t i, j, k;

  REALTYPEVEC shapeIntegral[3][3][2];

  __local REALTYPEVEC localResult[WORKGROUP_SIZE][3][2];
  REALTYPEVEC gradKernelValue[3][2];

  REALTYPEVEC myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS][2];

  REALTYPEVEC edgeLengths[3];

  REALTYPE3 evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);

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

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegral[i][j][0] = M_ZERO;
      shapeIntegral[i][j][1] = M_ZERO;
    }

  getCornersVec(grid, elementIndex, corners);
  getJacobianVec(corners, jacobian);
  getNormalAndIntegrationElementVec(jacobian, normal, &intElem);

  updateNormalsVec(elementIndex, normalSigns, normal);
  computeEdgeLengthVec(corners, edgeLengths);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point =
        (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    getGlobalPointVec(corners, &point, surfaceGlobalPoint);
    BASIS(SHAPESET, evaluate)(&point, &basisValue[0][0]);
    getPiolaTransformVec(intElem, jacobian, basisValue, elementValue);

    KERNEL_EXPLICIT(helmholtz_gradient, VEC_STRING)
    (evalGlobalPoint, surfaceGlobalPoint, testNormal, normal, kernel_parameters,
     gradKernelValue);

    for (i = 0; i < 3; ++i)
      for (k = 0; k < 2; ++k) {
        shapeIntegral[i][0][k] += (gradKernelValue[1][k] * elementValue[i][2] -
                                   gradKernelValue[2][k] * elementValue[i][1]) *
                                  quadWeights[quadIndex];
        shapeIntegral[i][1][k] += (gradKernelValue[2][k] * elementValue[i][0] -
                                   gradKernelValue[0][k] * elementValue[i][2]) *
                                  quadWeights[quadIndex];
        shapeIntegral[i][2][k] += (gradKernelValue[0][k] * elementValue[i][1] -
                                   gradKernelValue[1][k] * elementValue[i][0]) *
                                  quadWeights[quadIndex];
      }
  }

  for (j = 0; j < 3; ++j) {
    factor1[0] = M_ZERO;
    factor1[1] = M_ZERO;
    for (i = 0; i < 3; ++i) {
      factor1[0] += CMP_MULT_REAL(shapeIntegral[i][j], myCoefficients[i]) *
                    edgeLengths[i];
      factor1[1] += CMP_MULT_IMAG(shapeIntegral[i][j], myCoefficients[i]) *
                    edgeLengths[i];
    }
    localResult[lid][j][0] = factor1[0] * intElem;
    localResult[lid][j][1] = factor1[1] * intElem;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (i = 1; i < WORKGROUP_SIZE; ++i)
      for (j = 0; j < 3; ++j) {
        localResult[0][j][0] += localResult[i][j][0];
        localResult[0][j][1] += localResult[i][j][1];
      }
    for (int vecIndex = 0; vecIndex < VEC_LENGTH; ++vecIndex)
      for (j = 0; j < 3; ++j) {
        globalResult[2 * ((3 * gid[0] + j) * numGroups + groupId)] +=
            ((__local REALTYPE *)(&localResult[0][j][0]))[vecIndex];
        globalResult[2 * ((3 * gid[0] + j) * numGroups + groupId) + 1] +=
            ((__local REALTYPE *)(&localResult[0][j][1]))[vecIndex];
      }
  }
}
