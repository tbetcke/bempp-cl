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
    __global REALTYPE* kernel_parameters){
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

  REALTYPE testInv[2][2];
  REALTYPE trialInv[2][2];

  REALTYPE3 trialCurl[3];
  REALTYPE3 testCurl[3];

  REALTYPE basisProduct[3][3];
  REALTYPE normalProduct;

#ifndef COMPLEX_KERNEL
  __local REALTYPE localResult[WORKGROUP_SIZE][NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                              [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
  REALTYPE result[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                 [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
  REALTYPE kernelValue;
  REALTYPE firstTermIntegral;
#else
  __local REALTYPE localResult[WORKGROUP_SIZE][NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                              [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPE result[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                 [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPE kernelValue[2];
  REALTYPE firstTermIntegral[2];
  REALTYPE wavenumberProduct[2];
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

#ifndef COMPLEX_KERNEL
  firstTermIntegral = M_ZERO;
#else
  firstTermIntegral[0] = M_ZERO;
  firstTermIntegral[1] = M_ZERO;
#endif

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
                            testJac[1] * (-testInv[1][0] - testInv[1][1])) /
      (testIntElem * testIntElem);
  testCurl[1] = cross(testNormal,
                      testJac[0] * testInv[0][0] + testJac[1] * testInv[1][0]) /
                (testIntElem * testIntElem);
  testCurl[2] = cross(testNormal,
                      testJac[0] * testInv[0][1] + testJac[1] * testInv[1][1]) /
                (testIntElem * testIntElem);

  trialCurl[0] =
      cross(trialNormal, trialJac[0] * (-trialInv[0][0] - trialInv[0][1]) +
                             trialJac[1] * (-trialInv[1][0] - trialInv[1][1])) /
      (trialIntElem * trialIntElem);
  trialCurl[1] = cross(trialNormal, trialJac[0] * trialInv[0][0] +
                                        trialJac[1] * trialInv[1][0]) /
                 (trialIntElem * trialIntElem);
  trialCurl[2] = cross(trialNormal, trialJac[0] * trialInv[0][1] +
                                        trialJac[1] * trialInv[1][1]) /
                 (trialIntElem * trialIntElem);

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) basisProduct[i][j] = dot(testCurl[i], trialCurl[j]);

  normalProduct = dot(testNormal, trialNormal);

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
    firstTermIntegral += weight * kernelValue;
#else
    KERNEL(novec)
    (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters, kernelValue);
    firstTermIntegral[0] += weight * kernelValue[0];
    firstTermIntegral[1] += weight * kernelValue[1];

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
      localResult[localId][i][j] =
          (firstTermIntegral * basisProduct[i][j] +
           kernel_parameters[0] * kernel_parameters[0] * result[i][j] * normalProduct) *
          testIntElem * trialIntElem;
#else

  wavenumberProduct[0] = kernel_parameters[0] * kernel_parameters[0] -
                         kernel_parameters[1] * kernel_parameters[1];
  wavenumberProduct[1] = M_TWO * kernel_parameters[0] * kernel_parameters[1];

      localResult[localId][i][j][0] =
          (firstTermIntegral[0] * basisProduct[i][j] -
           normalProduct * (wavenumberProduct[0] * result[i][j][0] -
                            wavenumberProduct[1] * result[i][j][1])) *
          testIntElem * trialIntElem;
      localResult[localId][i][j][1] =
          (firstTermIntegral[1] * basisProduct[i][j] -
           normalProduct * (wavenumberProduct[0] * result[i][j][1] +
                            wavenumberProduct[1] * result[i][j][0])) *
          testIntElem * trialIntElem;
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
