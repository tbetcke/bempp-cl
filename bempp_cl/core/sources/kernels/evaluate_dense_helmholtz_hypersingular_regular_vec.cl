#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPEVEC))) void kernel_function(
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
  size_t offset = get_global_offset(1);

  size_t testIndex = testIndices[gid[0]];
  /* Macro to assign trial indices to new array trialIndex[VEC_LENGTH] */
  DEFINE_TRIAL_INDICES_REGULAR_ASSEMBLY

  size_t testQuadIndex;
  size_t trialQuadIndex;
  size_t i;
  size_t j;
  size_t k;
  size_t globalRowIndex;
  size_t globalColIndex;

  REALTYPE3 testGlobalPoint;
  REALTYPEVEC trialGlobalPoint[3];

  REALTYPE3 testCorners[3];
  REALTYPEVEC trialCorners[3][3];

  uint testElement[3];
  uint trialElement[VEC_LENGTH][3];

  uint myTestLocal2Global[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  uint myTrialLocal2Global[VEC_LENGTH][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE myTestLocalMultipliers[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  REALTYPE myTrialLocalMultipliers[VEC_LENGTH][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE3 testJac[2];
  REALTYPEVEC trialJac[2][3];

  REALTYPE3 testNormal;
  REALTYPEVEC trialNormal[3];

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;

  REALTYPE testIntElem;
  REALTYPEVEC trialIntElem;
  REALTYPE testValue[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  REALTYPE trialValue[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE testInv[2][2];
  REALTYPEVEC trialInv[2][2];

  REALTYPEVEC trialCurl[3][3];
  REALTYPE3 testCurl[3];

  REALTYPEVEC basisProduct[3][3];

  REALTYPEVEC trialInvBasis[2][3];
  REALTYPEVEC trialElementGradient[3][3];

  REALTYPEVEC normalProduct;

#ifndef COMPLEX_KERNEL
  REALTYPEVEC kernelValue;
  REALTYPEVEC tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
  REALTYPEVEC tempFactor;
  REALTYPEVEC shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                         [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPEVEC firstTermIntegral;
  REALTYPEVEC tempFirstTerm;

#else
  REALTYPEVEC tmp[2];
  REALTYPEVEC kernelValue[2];
  REALTYPEVEC tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPEVEC tempFactor[2];
  REALTYPEVEC shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                         [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPEVEC firstTermIntegral[2];
  REALTYPEVEC tempFirstTerm[2];
  REALTYPE wavenumberProduct[2];

#endif

#ifndef COMPLEX_KERNEL
  firstTermIntegral = M_ZERO;
#else
  firstTermIntegral[0] = M_ZERO;
  firstTermIntegral[1] = M_ZERO;
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

  getCorners(testGrid, testIndex, testCorners);
  getCornersVec(trialGrid, trialIndex, trialCorners);

  getElement(testConnectivity, testIndex, testElement);
  getElementVec(trialConnectivity, trialIndex, trialElement);

  getLocal2Global(testLocal2Global, testIndex, myTestLocal2Global,
                  NUMBER_OF_TEST_SHAPE_FUNCTIONS);
  getLocal2GlobalVec(trialLocal2Global, trialIndex, &myTrialLocal2Global[0][0],
                      NUMBER_OF_TRIAL_SHAPE_FUNCTIONS);

  getLocalMultipliers(testLocalMultipliers, testIndex, myTestLocalMultipliers,
                      NUMBER_OF_TEST_SHAPE_FUNCTIONS);
  getLocalMultipliersVec(trialLocalMultipliers, trialIndex,
                          &myTrialLocalMultipliers[0][0],
                          NUMBER_OF_TRIAL_SHAPE_FUNCTIONS);

  getJacobian(testCorners, testJac);
  getJacobianVec(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElementVec(trialJac, trialNormal, &trialIntElem);

  updateNormals(testIndex, testNormalSigns, &testNormal);
  updateNormalsVec(trialIndex, trialNormalSigns, trialNormal);

  testInv[0][0] = dot(testJac[1], testJac[1]);
  testInv[1][1] = dot(testJac[0], testJac[0]);
  testInv[0][1] = -dot(testJac[0], testJac[1]);
  testInv[1][0] = testInv[0][1];

  trialInv[0][0] = M_ZERO;
  trialInv[1][1] = M_ZERO;
  trialInv[0][1] = M_ZERO;

  for (i = 0; i < 3; ++i) {
    trialInv[0][0] += trialJac[1][i] * trialJac[1][i];
    trialInv[0][1] -= trialJac[0][i] * trialJac[1][i];
    trialInv[1][1] += trialJac[0][i] * trialJac[0][i];
  }

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

  for (i = 0; i < 2; ++i) {
    trialInvBasis[i][0] = -trialInv[i][0] - trialInv[i][1];
    trialInvBasis[i][1] = trialInv[i][0];
    trialInvBasis[i][2] = trialInv[i][1];
  }

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      trialElementGradient[j][i] = 0;
      for (k = 0; k < 2; ++k)
        trialElementGradient[j][i] += trialJac[k][i] * trialInvBasis[k][j];
    }

  for (i = 0; i < 3; ++i) {
    trialCurl[i][0] = (trialNormal[1] * trialElementGradient[i][2] -
                       trialNormal[2] * trialElementGradient[i][1]) /
                      (trialIntElem * trialIntElem);
    trialCurl[i][1] = (trialNormal[2] * trialElementGradient[i][0] -
                       trialNormal[0] * trialElementGradient[i][2]) /
                      (trialIntElem * trialIntElem);
    trialCurl[i][2] = (trialNormal[0] * trialElementGradient[i][1] -
                       trialNormal[1] * trialElementGradient[i][0]) /
                      (trialIntElem * trialIntElem);
  }

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      basisProduct[i][j] = testCurl[i].x * trialCurl[j][0] +
                           testCurl[i].y * trialCurl[j][1] +
                           testCurl[i].z * trialCurl[j][2];
    }

  normalProduct = testNormal.x * trialNormal[0] +
                  testNormal.y * trialNormal[1] + testNormal.z * trialNormal[2];

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
      getGlobalPointVec(trialCorners, &trialPoint, trialGlobalPoint);
      BASIS(TRIAL, evaluate)(&trialPoint, &trialValue[0]);
#ifndef COMPLEX_KERNEL
      KERNEL(VEC_STRING)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters,
       &kernelValue);
      tempFactor = quadWeights[trialQuadIndex] * kernelValue;
      tempFirstTerm += tempFactor;
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
        tempResult[j] += trialValue[j] * tempFactor;
#else
      KERNEL(VEC_STRING)
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
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
        shapeIntegral[i][j] +=
            tempResult[j] * quadWeights[testQuadIndex] * testValue[i];
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
          (kernel_parameters[0] * kernel_parameters[0] * shapeIntegral[i][j] * normalProduct +
           firstTermIntegral * basisProduct[i][j]) *
          testIntElem * trialIntElem;

#else

  wavenumberProduct[0] = kernel_parameters[0] * kernel_parameters[0] -
                         kernel_parameters[1] * kernel_parameters[1];
  wavenumberProduct[1] = M_TWO * kernel_parameters[0] * kernel_parameters[1];

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      tmp[0] = shapeIntegral[i][j][0];
      tmp[1] = shapeIntegral[i][j][1];
      shapeIntegral[i][j][0] =
          (-(wavenumberProduct[0] * tmp[0] -
             wavenumberProduct[1] * tmp[1]) *
               normalProduct +
           firstTermIntegral[0] * basisProduct[i][j]) *
          testIntElem * trialIntElem;
      shapeIntegral[i][j][1] =
          (-(wavenumberProduct[0] * tmp[1] +
             wavenumberProduct[1] * tmp[0]) *
               normalProduct +
           firstTermIntegral[1] * basisProduct[i][j]) *
          testIntElem * trialIntElem;
    }

#endif

  for (int vecIndex = 0; vecIndex < VEC_LENGTH; ++vecIndex)
    if (!elementsAreAdjacent(testElement, trialElement[vecIndex],
                             gridsAreDisjoint)) {
      for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
        for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
          globalRowIndex = myTestLocal2Global[i];
          globalColIndex = myTrialLocal2Global[vecIndex][j];
#ifndef COMPLEX_KERNEL
          globalResult[globalRowIndex * nTrial + globalColIndex] +=
              ((REALTYPE*)(&shapeIntegral[i][j]))[vecIndex] *
              myTestLocalMultipliers[i] * myTrialLocalMultipliers[vecIndex][j];
#else
          globalResult[2 * (globalRowIndex * nTrial + globalColIndex)] +=
              ((REALTYPE*)(&shapeIntegral[i][j][0]))[vecIndex] *
              myTestLocalMultipliers[i] * myTrialLocalMultipliers[vecIndex][j];
          globalResult[2 * (globalRowIndex * nTrial + globalColIndex) + 1] +=
              ((REALTYPE*)(&shapeIntegral[i][j][1]))[vecIndex] *
              myTestLocalMultipliers[i] * myTrialLocalMultipliers[vecIndex][j];
#endif
        }
    }
}
