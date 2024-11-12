#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void kernel_function(
    __global uint* testIndices, __global uint* trialIndices,
    __global int *testNormalSigns, __global int *trialNormalSigns,
    __global REALTYPE* testGrid, __global REALTYPE* trialGrid,
    __global uint* testConnectivity, __global uint* trialConnectivity,
    __global uint* testLocal2Global, __global uint* trialLocal2Global,
    __global REALTYPE* testLocalMultipliers,
    __global REALTYPE* trialLocalMultipliers, __constant REALTYPE* quadPoints,
    __constant REALTYPE* quadWeights, __global REALTYPE* globalResult,
    __global REALTYPE* kernel_parameters,
    int nTest, int nTrial, char gridsAreDisjoint) {
  /* Variable declarations */

  size_t gid[2] = {get_global_id(0), get_global_id(1)};

  size_t testIndex = testIndices[gid[0]];
  size_t trialIndex = trialIndices[gid[1]];

  size_t testQuadIndex;
  size_t trialQuadIndex;
  size_t i;
  size_t j;
  size_t globalRowIndex;
  size_t globalColIndex;

  REALTYPE3 testGlobalPoint;
  REALTYPE3 trialGlobalPoint;

  REALTYPE3 testCorners[3];
  REALTYPE3 trialCorners[3];

  uint testElement[3];
  uint trialElement[3];

  uint myTestLocal2Global[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  uint myTrialLocal2Global[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE myTestLocalMultipliers[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  REALTYPE myTrialLocalMultipliers[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE3 testJac[2];
  REALTYPE3 trialJac[2];

  REALTYPE3 testNormal;
  REALTYPE3 trialNormal;

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;

  REALTYPE testIntElem;
  REALTYPE trialIntElem;
  REALTYPE testValue[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  REALTYPE trialValue[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE testInv[2][2];
  REALTYPE trialInv[2][2];

  REALTYPE3 trialCurl[3];
  REALTYPE3 testCurl[3];

  REALTYPE basisProduct[3][3];
  REALTYPE normalProduct;

#ifndef COMPLEX_KERNEL
  REALTYPE kernelValue;
  REALTYPE tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
  REALTYPE tempFactor;
  REALTYPE shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                        [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
  REALTYPE firstTermIntegral;
  REALTYPE tempFirstTerm;
#else
  REALTYPE tmp[2];
  REALTYPE kernelValue[2];
  REALTYPE tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPE tempFactor[2];
  REALTYPE shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                        [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPE firstTermIntegral[2];
  REALTYPE tempFirstTerm[2];
  REALTYPE wavenumberProduct[2];
#endif

  for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
      shapeIntegral[i][j] = M_ZERO;
#else
      shapeIntegral[i][j][0] = M_ZERO;
      shapeIntegral[i][j][1] = M_ZERO;
#endif
    }

#ifndef COMPLEX_KERNEL
  firstTermIntegral = M_ZERO;
#else
  firstTermIntegral[0] = M_ZERO;
  firstTermIntegral[1] = M_ZERO;
#endif

  getCorners(testGrid, testIndex, testCorners);
  getCorners(trialGrid, trialIndex, trialCorners);

  getElement(testConnectivity, testIndex, testElement);
  getElement(trialConnectivity, trialIndex, trialElement);

  getLocal2Global(testLocal2Global, testIndex, myTestLocal2Global,
                  NUMBER_OF_TEST_SHAPE_FUNCTIONS);
  getLocal2Global(trialLocal2Global, trialIndex, myTrialLocal2Global,
                  NUMBER_OF_TRIAL_SHAPE_FUNCTIONS);

  getJacobian(testCorners, testJac);
  getJacobian(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElement(trialJac, &trialNormal, &trialIntElem);

  getLocalMultipliers(testLocalMultipliers, testIndex, myTestLocalMultipliers,
                      NUMBER_OF_TEST_SHAPE_FUNCTIONS);
  getLocalMultipliers(trialLocalMultipliers, trialIndex,
                      myTrialLocalMultipliers, NUMBER_OF_TRIAL_SHAPE_FUNCTIONS);

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

  for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
       ++testQuadIndex) {
    testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex], quadPoints[2 * testQuadIndex + 1]);
    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
    BASIS(TEST, evaluate)(&testPoint, &testValue[0]);

#ifndef COMPLEX_KERNEL
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
      tempResult[j] = M_ZERO;
    }
    tempFirstTerm = M_ZERO;
#else
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
      tempResult[j][0] = M_ZERO;
      tempResult[j][1] = M_ZERO;
    }

    tempFirstTerm[0] = M_ZERO;
    tempFirstTerm[1] = M_ZERO;
#endif

    for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
         ++trialQuadIndex) {
      trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex], quadPoints[2 * trialQuadIndex + 1]);
      trialGlobalPoint = getGlobalPoint(trialCorners, &trialPoint);
      BASIS(TRIAL, evaluate)(&trialPoint, &trialValue[0]);
#ifndef COMPLEX_KERNEL
      KERNEL(novec)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters,
       &kernelValue);
      tempFactor = quadWeights[trialQuadIndex] * kernelValue;
      tempFirstTerm += tempFactor;
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
        tempResult[j] += trialValue[j] * tempFactor;
#else
      KERNEL(novec)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters, kernelValue);
      tempFactor[0] = quadWeights[trialQuadIndex] * kernelValue[0];
      tempFactor[1] = quadWeights[trialQuadIndex] * kernelValue[1];
      tempFirstTerm[0] += tempFactor[0];
      tempFirstTerm[1] += tempFactor[1];
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
        tempResult[j][0] += trialValue[j] * tempFactor[0];
        tempResult[j][1] += trialValue[j] * tempFactor[1];
      }

#endif
    }

#ifndef COMPLEX_KERNEL
    firstTermIntegral += tempFirstTerm * quadWeights[testQuadIndex];
    for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
        shapeIntegral[i][j] +=
            tempResult[j] * quadWeights[testQuadIndex] * testValue[i];
      }
#else
    firstTermIntegral[0] += tempFirstTerm[0] * quadWeights[testQuadIndex];
    firstTermIntegral[1] += tempFirstTerm[1] * quadWeights[testQuadIndex];
    for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
        shapeIntegral[i][j][0] +=
            tempResult[j][0] * quadWeights[testQuadIndex] * testValue[i];
        shapeIntegral[i][j][1] +=
            tempResult[j][1] * quadWeights[testQuadIndex] * testValue[i];
      }
#endif
  }

#ifndef COMPLEX_KERNEL
  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      shapeIntegral[i][j] =
          kernel_parameters[0] * kernel_parameters[0] * shapeIntegral[i][j] * normalProduct +
          firstTermIntegral * basisProduct[i][j];

#else

  wavenumberProduct[0] = kernel_parameters[0] * kernel_parameters[0] -
                         kernel_parameters[1] * kernel_parameters[1];
  wavenumberProduct[1] = M_TWO * kernel_parameters[0] * kernel_parameters[1];

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      tmp[0] = shapeIntegral[i][j][0];
      tmp[1] = shapeIntegral[i][j][1];
      shapeIntegral[i][j][0] =
          -(wavenumberProduct[0] * tmp[0] -
            wavenumberProduct[1] * tmp[1]) *
              normalProduct +
          firstTermIntegral[0] * basisProduct[i][j];
      shapeIntegral[i][j][1] =
          -(wavenumberProduct[0] * tmp[1] +
            wavenumberProduct[1] * tmp[0]) *
              normalProduct +
          firstTermIntegral[1] * basisProduct[i][j];
    }

#endif

  if (!elementsAreAdjacent(testElement, trialElement, gridsAreDisjoint)) {
    for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
        globalRowIndex = myTestLocal2Global[i];
        globalColIndex = myTrialLocal2Global[j];
#ifndef COMPLEX_KERNEL
        globalResult[globalRowIndex * nTrial + globalColIndex] +=
            shapeIntegral[i][j] * testIntElem * trialIntElem *
            myTestLocalMultipliers[i] * myTrialLocalMultipliers[j];
#else
        globalResult[2 * (globalRowIndex * nTrial + globalColIndex)] +=
            shapeIntegral[i][j][0] * testIntElem * trialIntElem *
            myTestLocalMultipliers[i] * myTrialLocalMultipliers[j];
        globalResult[2 * (globalRowIndex * nTrial + globalColIndex) + 1] +=
            shapeIntegral[i][j][1] * testIntElem * trialIntElem *
            myTestLocalMultipliers[i] * myTrialLocalMultipliers[j];
#endif
      }
  }
}
