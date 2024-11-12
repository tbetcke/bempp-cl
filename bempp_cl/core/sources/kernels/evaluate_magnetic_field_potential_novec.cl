#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void kernel_function(
    __global REALTYPE *grid, 
    __global uint* indices,
    __global int* normalSigns,
    __global REALTYPE *evalPoints,
    __global REALTYPE *coefficients, __constant REALTYPE *quadPoints,
    __constant REALTYPE *quadWeights, __global REALTYPE *globalResult,
    __global REALTYPE* kernel_parameters) {
  size_t gid[2];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);

  size_t elementIndex = indices[gid[1]];

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);

  REALTYPE3 surfaceGlobalPoint;

  REALTYPE basisValue[3][2];
  REALTYPE3 elementValue[3];

  REALTYPE3 corners[3];
  REALTYPE3 jacobian[2];
  REALTYPE3 normal;

  REALTYPE factor1[2];

  REALTYPE2 point;

  REALTYPE intElem;

  size_t quadIndex;
  size_t i, j, k;

  REALTYPE shapeIntegral[3][3][2];

  __local REALTYPE localResult[WORKGROUP_SIZE][3][2];
  REALTYPE gradKernelValue[3][2];
  REALTYPE myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS][2];
  REALTYPE edgeLengths[3];

  REALTYPE3 evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);

#ifndef COMPLEX_COEFFICIENTS
  for (i = 0; i < 3; ++i) {
    myCoefficients[i][0] = coefficients[3 * elementIndex + i];
    myCoefficients[i][1] = M_ZERO;
  }
#else
  for (i = 0; i < 3; ++i) {
    myCoefficients[i][0] = coefficients[2 * (3 * elementIndex + i)];
    myCoefficients[i][1] = coefficients[2 * (3 * elementIndex + i) + 1];
  }
#endif

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegral[i][j][0] = M_ZERO;
      shapeIntegral[i][j][1] = M_ZERO;
    }

  getCorners(grid, elementIndex, corners);
  getJacobian(corners, jacobian);
  getNormalAndIntegrationElement(jacobian, &normal, &intElem);

  updateNormals(elementIndex, normalSigns, &normal);
  computeEdgeLength(corners, edgeLengths);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point =
        (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    surfaceGlobalPoint = getGlobalPoint(corners, &point);
    BASIS(SHAPESET, evaluate)(&point, &basisValue[0][0]);
    getPiolaTransform(intElem, jacobian, basisValue, elementValue);

    helmholtz_gradient_novec(evalGlobalPoint, surfaceGlobalPoint, normal, normal, kernel_parameters, gradKernelValue); 


    for (i = 0; i < 3; ++i)
        for (k = 0; k < 2; ++k){
        shapeIntegral[i][0][k] += (gradKernelValue[1][k] * elementValue[i].z - gradKernelValue[2][k] * elementValue[i].y)  * quadWeights[quadIndex];
        shapeIntegral[i][1][k] += (gradKernelValue[2][k] * elementValue[i].x - gradKernelValue[0][k] * elementValue[i].z)  * quadWeights[quadIndex];
        shapeIntegral[i][2][k] += (gradKernelValue[0][k] * elementValue[i].y - gradKernelValue[1][k] * elementValue[i].x)  * quadWeights[quadIndex];
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
    for (j = 0; j < 3; ++j) {
      globalResult[2 * ((3 * gid[0] + j) * numGroups +
                        groupId)] += localResult[0][j][0];
      globalResult[2 * ((3 * gid[0] + j) * numGroups + groupId) +
                   1] += localResult[0][j][1];
    }
  }
}
