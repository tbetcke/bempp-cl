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

#ifndef COMPLEX_KERNEL
  REALTYPEVEC kernelValue;
  REALTYPEVEC tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
  REALTYPEVEC tempFactor;
  REALTYPEVEC shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                         [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
#else
  REALTYPEVEC kernelValue[2];
  REALTYPEVEC tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPEVEC tempFactor[2];
  REALTYPEVEC shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                         [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
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

  for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
       ++testQuadIndex) {
    testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex], quadPoints[2 * testQuadIndex + 1]);
    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
    BASIS(TEST, evaluate)(&testPoint, &testValue[0]);

    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
      tempResult[j] = M_ZERO;
#else
      tempResult[j][0] = M_ZERO;
      tempResult[j][1] = M_ZERO;
#endif
    }

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
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
        tempResult[j] += trialValue[j] * tempFactor;
#else
      KERNEL(VEC_STRING)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters, kernelValue);
      tempFactor[0] = quadWeights[trialQuadIndex] * kernelValue[0];
      tempFactor[1] = quadWeights[trialQuadIndex] * kernelValue[1];
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
        tempResult[j][0] += trialValue[j] * tempFactor[0];
        tempResult[j][1] += trialValue[j] * tempFactor[1];
      }

#endif
    }

    for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
        shapeIntegral[i][j] +=
            tempResult[j] * quadWeights[testQuadIndex] * testValue[i];
#else
        shapeIntegral[i][j][0] +=
            tempResult[j][0] * quadWeights[testQuadIndex] * testValue[i];
        shapeIntegral[i][j][1] +=
            tempResult[j][1] * quadWeights[testQuadIndex] * testValue[i];
#endif
      }
  }

  for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
      shapeIntegral[i][j] *=
          (testIntElem * myTestLocalMultipliers[i]) * trialIntElem;
#else
      shapeIntegral[i][j][0] *=
          (testIntElem * myTestLocalMultipliers[i]) * trialIntElem;
      shapeIntegral[i][j][1] *=
          (testIntElem * myTestLocalMultipliers[i]) * trialIntElem;
#endif
    }

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
              myTrialLocalMultipliers[vecIndex][j];
#else
          globalResult[2 * (globalRowIndex * nTrial + globalColIndex)] +=
              ((REALTYPE*)(&shapeIntegral[i][j][0]))[vecIndex] *
              myTrialLocalMultipliers[vecIndex][j];
          globalResult[2 * (globalRowIndex * nTrial + globalColIndex) + 1] +=
              ((REALTYPE*)(&shapeIntegral[i][j][1]))[vecIndex] *
              myTrialLocalMultipliers[vecIndex][j];
#endif
        }
    }
}
