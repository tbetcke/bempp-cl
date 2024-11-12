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
    __global REALTYPE *globalResult,
    __global REALTYPE *kernel_parameters) {
  /* Variable declarations */

  size_t groupId;
  size_t localId;

  int i, j, m;

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;
  REALTYPE weight;
  REALTYPE dist;

  REALTYPE3 testGlobalPoint;
  REALTYPE3 trialGlobalPoint;

  REALTYPE testValue[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  REALTYPE trialValue[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE3 testCorners[3];
  REALTYPE3 trialCorners[3];

  REALTYPE3 testNormal;
  REALTYPE3 trialNormal;

  REALTYPE3 testJac[2];
  REALTYPE3 trialJac[2];

  REALTYPE testIntElem;
  REALTYPE trialIntElem;

  REALTYPE testInv[2][2];
  REALTYPE trialInv[2][2];

  REALTYPE3 trialCurl[3];
  REALTYPE3 testCurl[3];

  REALTYPE basisProduct[3][3];

  uint localQuadPointsPerItem;

  uint localTestOffset;
  uint localTrialOffset;
  uint localWeightsOffset;

  uint testIndex;
  uint trialIndex;

  __local REALTYPE localResult[WORKGROUP_SIZE];
  REALTYPE result;
  REALTYPE kernelValue;

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

  updateNormals(testIndex, testNormalSigns, &testNormal);
  updateNormals(trialIndex, trialNormalSigns, &trialNormal);

  testInv[0][0] = dot(testJac[1], testJac[1]);
  testInv[1][1] = dot(testJac[0], testJac[0]);
  testInv[0][1] = -dot(testJac[0], testJac[1]);
  testInv[1][0] = testInv[0][1];

  trialInv[0][0] = dot(trialJac[1], trialJac[1]);
  trialInv[1][1] = dot(trialJac[0], trialJac[0]);
  trialInv[0][1] = -dot(trialJac[0], trialJac[1]);
  trialInv[1][0] = trialInv[0][1];

  testCurl[0] =
      cross(testNormal, testJac[0] * (-testInv[0][0] - testInv[0][1]) +
                            testJac[1] * (-testInv[1][0] - testInv[1][1]));
  testCurl[1] = cross(testNormal,
                      testJac[0] * testInv[0][0] + testJac[1] * testInv[1][0]);
  testCurl[2] = cross(testNormal,
                      testJac[0] * testInv[0][1] + testJac[1] * testInv[1][1]);

  trialCurl[0] =
      cross(trialNormal, trialJac[0] * (-trialInv[0][0] - trialInv[0][1]) +
                             trialJac[1] * (-trialInv[1][0] - trialInv[1][1]));
  trialCurl[1] = cross(
      trialNormal, trialJac[0] * trialInv[0][0] + trialJac[1] * trialInv[1][0]);
  trialCurl[2] = cross(
      trialNormal, trialJac[0] * trialInv[0][1] + trialJac[1] * trialInv[1][1]);

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) basisProduct[i][j] = dot(testCurl[i], trialCurl[j]);

  result = M_ZERO;

  for (uint quadIndex = localQuadPointsPerItem * localId;
       quadIndex < localQuadPointsPerItem * (localId + 1); ++quadIndex) {
    testPoint = (REALTYPE2)(testPoints[2 * (localTestOffset + quadIndex)], testPoints[2 * (localTestOffset + quadIndex) + 1]);
    trialPoint = (REALTYPE2)(trialPoints[2 * (localTrialOffset + quadIndex)], trialPoints[2 * (localTrialOffset + quadIndex) + 1]);
    weight = quadWeights[localWeightsOffset + quadIndex];

    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
    trialGlobalPoint = getGlobalPoint(trialCorners, &trialPoint);

    KERNEL(novec)
    (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters, &kernelValue);

    result += weight * kernelValue;
  }

  // the Jacobian Inverse must by divded by the squared of the integration
  // elements. The integral must be multiplied by the integration elements. So
  // in total we have to divide once.

  localResult[localId] = result / (testIntElem * trialIntElem);

  barrier(CLK_LOCAL_MEM_FENCE);

  if (localId == 0) {
    for (m = 1; m < WORKGROUP_SIZE; ++m) localResult[0] += localResult[m];
    for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
        globalResult[NUMBER_OF_TEST_SHAPE_FUNCTIONS *
                         NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * groupId +
                     i * NUMBER_OF_TRIAL_SHAPE_FUNCTIONS + j] =
            localResult[0] * basisProduct[i][j];
      }
  }
}
