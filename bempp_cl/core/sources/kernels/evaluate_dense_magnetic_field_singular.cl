#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void kernel_function(
    __global REALTYPE *grid,
    __global int *testNormalSigns, __global int *trialNormalSigns,
    __global REALTYPE *testPoints,
    __global REALTYPE *trialPoints, __global REALTYPE *quadWeights,
    __global uint *testIndices, __global uint *trialIndices,
    __global uint *testOffsets, __global uint *trialOffsets,
    __global uint *weightOffsets, __global uint *numberOfLocalQuadPoints,
    __global REALTYPE* globalResult,
    __global REALTYPE *kernel_parameters) {
  /* Variable declarations */

  size_t groupId;
  size_t localId;

  int i, j, m, k;

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;
  REALTYPE weight;
  REALTYPE dist;

  REALTYPE3 testGlobalPoint;
  REALTYPE3 trialGlobalPoint;

  REALTYPE testValue[3][2];
  REALTYPE trialValue[3][2];

  REALTYPE3 testElementValue[3];
  REALTYPE3 trialElementValue[3];
  REALTYPE testEdgeLength[3];
  REALTYPE trialEdgeLength[3];

  REALTYPE3 testCorners[3];
  REALTYPE3 trialCorners[3];

  REALTYPE3 testNormal;
  REALTYPE3 trialNormal;

  REALTYPE3 testJac[2];
  REALTYPE3 trialJac[2];

  REALTYPE testIntElem;
  REALTYPE trialIntElem;

  REALTYPE shapeIntegral[3][3][2];
  REALTYPE kernelValue[3][2];

  uint localQuadPointsPerItem;

  uint localTestOffset;
  uint localTrialOffset;
  uint localWeightsOffset;

  uint testIndex;
  uint trialIndex;

  __local REALTYPE localResult[WORKGROUP_SIZE][3][3][2];

  groupId = get_group_id(0);
  localId = get_local_id(0);

  localQuadPointsPerItem = numberOfLocalQuadPoints[groupId];
  localTestOffset = testOffsets[groupId];
  localTrialOffset = trialOffsets[groupId];
  localWeightsOffset = weightOffsets[groupId];
  testIndex = testIndices[groupId];
  trialIndex = trialIndices[groupId];

  getCorners(grid, testIndex, testCorners);
  getCorners(grid, trialIndex, trialCorners);

  getJacobian(testCorners, testJac);
  getJacobian(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElement(trialJac, &trialNormal, &trialIntElem);

  computeEdgeLength(testCorners, testEdgeLength);
  computeEdgeLength(trialCorners, trialEdgeLength);

  updateNormals(testIndex, testNormalSigns, &testNormal);
  updateNormals(trialIndex, trialNormalSigns, &trialNormal);

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegral[i][j][0] = M_ZERO;
      shapeIntegral[i][j][1] = M_ZERO;
    }

  for (uint quadIndex = localQuadPointsPerItem * localId;
       quadIndex < localQuadPointsPerItem * (localId + 1); ++quadIndex) {
    testPoint = (REALTYPE2)(testPoints[2 * (localTestOffset + quadIndex)], testPoints[2 * (localTestOffset + quadIndex) + 1]);
    trialPoint = (REALTYPE2)(trialPoints[2 * (localTrialOffset + quadIndex)], trialPoints[2 * (localTrialOffset + quadIndex) + 1]);
    weight = quadWeights[localWeightsOffset + quadIndex];
    BASIS(TEST, evaluate)
    (&testPoint, &testValue[0][0]);
    BASIS(TRIAL, evaluate)
    (&trialPoint, &trialValue[0][0]);
    getPiolaTransform(testIntElem, testJac, testValue, testElementValue);
    getPiolaTransform(trialIntElem, trialJac, trialValue, trialElementValue);

    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
    trialGlobalPoint = getGlobalPoint(trialCorners, &trialPoint);

    helmholtz_gradient_novec(testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters, kernelValue);

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
        for (k = 0; k < 2; ++k)
          shapeIntegral[i][j][k] -=
              weight * (testElementValue[i].x *
                            (kernelValue[1][k] * trialElementValue[j].z -
                             kernelValue[2][k] * trialElementValue[j].y) +
                        testElementValue[i].y *
                            (kernelValue[2][k] * trialElementValue[j].x -
                             kernelValue[0][k] * trialElementValue[j].z) +
                        testElementValue[i].z *
                            (kernelValue[0][k] * trialElementValue[j].y -
                             kernelValue[1][k] * trialElementValue[j].x));
  }

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      localResult[localId][i][j][0] = shapeIntegral[i][j][0] * testIntElem *
                                      trialIntElem * testEdgeLength[i] *
                                      trialEdgeLength[j];
      localResult[localId][i][j][1] = shapeIntegral[i][j][1] * testIntElem *
                                      trialIntElem * testEdgeLength[i] *
                                      trialEdgeLength[j];
    }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (localId == 0) {
    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j) {
        for (m = 1; m < WORKGROUP_SIZE; ++m) {
          localResult[0][i][j][0] += localResult[m][i][j][0];
          localResult[0][i][j][1] += localResult[m][i][j][1];
        }
        globalResult[2 * (9 * groupId + i * 3 + j)] = localResult[0][i][j][0];
        globalResult[2 * (9 * groupId + i * 3 + j) + 1] =
            localResult[0][i][j][1];
      }
  }
}
