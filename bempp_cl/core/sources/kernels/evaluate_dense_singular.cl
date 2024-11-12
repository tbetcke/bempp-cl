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

  uint localQuadPointsPerItem;

  uint localTestOffset;
  uint localTrialOffset;
  uint localWeightsOffset;

  uint testIndex;
  uint trialIndex;

#ifndef COMPLEX_KERNEL
  __local REALTYPE localResult[WORKGROUP_SIZE][NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                              [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
  REALTYPE result[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                 [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
  REALTYPE kernelValue;
#else
  __local REALTYPE localResult[WORKGROUP_SIZE][NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                              [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPE result[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                 [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPE kernelValue[2];
#endif

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

  for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
      result[i][j] = M_ZERO;
#else
      result[i][j][0] = M_ZERO;
      result[i][j][1] = M_ZERO;
#endif
    }

  for (uint quadIndex = localQuadPointsPerItem * localId;
       quadIndex < localQuadPointsPerItem * (localId + 1); ++quadIndex) {
    testPoint = (REALTYPE2)(testPoints[2 * (localTestOffset + quadIndex)], testPoints[2 * (localTestOffset + quadIndex) + 1]);
    trialPoint = (REALTYPE2)(trialPoints[2 * (localTrialOffset + quadIndex)], trialPoints[2 * (localTrialOffset + quadIndex) + 1]);
    weight = quadWeights[localWeightsOffset + quadIndex];
    BASIS(TEST, evaluate)(&testPoint, &testValue[0]);
    BASIS(TRIAL, evaluate)(&trialPoint, &trialValue[0]);

    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
    trialGlobalPoint = getGlobalPoint(trialCorners, &trialPoint);

#ifndef COMPLEX_KERNEL
    KERNEL(novec)
    (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters, &kernelValue);
#else
    KERNEL(novec)
    (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters, kernelValue);
#endif

    for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
        result[i][j] += testValue[i] * trialValue[j] * weight * kernelValue;
#else
        result[i][j][0] +=
            testValue[i] * trialValue[j] * weight * kernelValue[0];
        result[i][j][1] +=
            testValue[i] * trialValue[j] * weight * kernelValue[1];
#endif
      }
  }

  for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
      localResult[localId][i][j] = result[i][j] * testIntElem * trialIntElem;
#else
      localResult[localId][i][j][0] =
          result[i][j][0] * testIntElem * trialIntElem;
      localResult[localId][i][j][1] =
          result[i][j][1] * testIntElem * trialIntElem;
#endif
    }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (localId == 0) {
    for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
        for (m = 1; m < WORKGROUP_SIZE; ++m)
          localResult[0][i][j] += localResult[m][i][j];
        globalResult[NUMBER_OF_TEST_SHAPE_FUNCTIONS *
                         NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * groupId +
                     i * NUMBER_OF_TRIAL_SHAPE_FUNCTIONS + j] =
            localResult[0][i][j];
#else
        for (m = 1; m < WORKGROUP_SIZE; ++m) {
          localResult[0][i][j][0] += localResult[m][i][j][0];
          localResult[0][i][j][1] += localResult[m][i][j][1];
        }
        globalResult[2 * (NUMBER_OF_TEST_SHAPE_FUNCTIONS *
                              NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * groupId +
                          i * NUMBER_OF_TRIAL_SHAPE_FUNCTIONS + j)] =
            localResult[0][i][j][0];
        globalResult[2 * (NUMBER_OF_TEST_SHAPE_FUNCTIONS *
                              NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * groupId +
                          i * NUMBER_OF_TRIAL_SHAPE_FUNCTIONS + j) +
                     1] = localResult[0][i][j][1];
#endif
      }
  }
}
