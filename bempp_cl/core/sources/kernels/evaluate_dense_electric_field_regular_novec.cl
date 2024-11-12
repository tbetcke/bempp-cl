#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void kernel_function(
    __global uint *testIndices, __global uint *trialIndices,
    __global int *testNormalSigns, __global int *trialNormalSigns,
    __global REALTYPE *testGrid, __global REALTYPE *trialGrid,
    __global uint *testConnectivity, __global uint *trialConnectivity,
    __global uint *testLocal2Global, __global uint *trialLocal2Global,
    __global REALTYPE *testLocalMultipliers,
    __global REALTYPE *trialLocalMultipliers, __constant REALTYPE *quadPoints,
    __constant REALTYPE *quadWeights, __global REALTYPE *globalResult,
    __global REALTYPE *kernel_parameters, int nTest, int nTrial,
    char gridsAreDisjoint) {
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

  uint myTestLocal2Global[3];
  uint myTrialLocal2Global[3];

  REALTYPE myTestLocalMultipliers[3];
  REALTYPE myTrialLocalMultipliers[3];

  REALTYPE3 testJac[2];
  REALTYPE3 trialJac[2];

  REALTYPE3 testNormal;
  REALTYPE3 trialNormal;

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;

  REALTYPE testIntElem;
  REALTYPE trialIntElem;
  REALTYPE testValue[3][2];
  REALTYPE trialValue[3][2];
  REALTYPE3 testElementValue[3];
  REALTYPE3 trialElementValue[3];
  REALTYPE testEdgeLength[3];
  REALTYPE trialEdgeLength[3];

  REALTYPE kernelValue[2];
  REALTYPE tempFactor[2];
  REALTYPE3 tempResultFirstComponent[3][2];
  REALTYPE tempResultSecondComponent[2];
  REALTYPE shapeIntegralFirstComponent[3][3][2];
  REALTYPE shapeIntegralSecondComponent[2];
  REALTYPE shiftedWavenumber[2] = {M_ZERO, M_ZERO};
  REALTYPE inverseShiftedWavenumber[2] = {M_ZERO, M_ZERO};
  REALTYPE divergenceProduct;

  REALTYPE shapeIntegral[3][3][2];

  // Computation of 1i * wavenumber and 1 / (1i * wavenumber)
  shiftedWavenumber[0] = -kernel_parameters[1];
  shiftedWavenumber[1] = kernel_parameters[0];

  inverseShiftedWavenumber[0] = M_ONE /
                                (shiftedWavenumber[0] * shiftedWavenumber[0] +
                                 shiftedWavenumber[1] * shiftedWavenumber[1]) *
                                shiftedWavenumber[0];
  inverseShiftedWavenumber[1] = -M_ONE /
                                (shiftedWavenumber[0] * shiftedWavenumber[0] +
                                 shiftedWavenumber[1] * shiftedWavenumber[1]) *
                                shiftedWavenumber[1];

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegralFirstComponent[i][j][0] = M_ZERO;
      shapeIntegralFirstComponent[i][j][1] = M_ZERO;
    }
  shapeIntegralSecondComponent[0] = M_ZERO;
  shapeIntegralSecondComponent[1] = M_ZERO;

  getCorners(testGrid, testIndex, testCorners);
  getCorners(trialGrid, trialIndex, trialCorners);

  getElement(testConnectivity, testIndex, testElement);
  getElement(trialConnectivity, trialIndex, trialElement);

  getLocal2Global(testLocal2Global, testIndex, myTestLocal2Global, 3);
  getLocal2Global(trialLocal2Global, trialIndex, myTrialLocal2Global, 3);

  getLocalMultipliers(testLocalMultipliers, testIndex, myTestLocalMultipliers,
                      3);
  getLocalMultipliers(trialLocalMultipliers, trialIndex,
                      myTrialLocalMultipliers, 3);

  getJacobian(testCorners, testJac);
  getJacobian(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElement(trialJac, &trialNormal, &trialIntElem);

  computeEdgeLength(testCorners, testEdgeLength);
  computeEdgeLength(trialCorners, trialEdgeLength);

  updateNormals(testIndex, testNormalSigns, &testNormal);
  updateNormals(trialIndex, trialNormalSigns, &trialNormal);

  for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
       ++testQuadIndex) {
    testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex],
                            quadPoints[2 * testQuadIndex + 1]);
    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
    BASIS(TEST, evaluate)
    (&testPoint, &testValue[0][0]);
    getPiolaTransform(testIntElem, testJac, testValue, testElementValue);

    for (j = 0; j < 3; ++j) {
      tempResultFirstComponent[j][0] = M_ZERO;
      tempResultFirstComponent[j][1] = M_ZERO;
    }
    tempResultSecondComponent[0] = M_ZERO;
    tempResultSecondComponent[1] = M_ZERO;

    for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
         ++trialQuadIndex) {
      trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex],
                               quadPoints[2 * trialQuadIndex + 1]);
      trialGlobalPoint = getGlobalPoint(trialCorners, &trialPoint);
      BASIS(TRIAL, evaluate)
      (&trialPoint, &trialValue[0][0]);
      getPiolaTransform(trialIntElem, trialJac, trialValue, trialElementValue);
      KERNEL(novec)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal,
       kernel_parameters, kernelValue);

      tempFactor[0] = kernelValue[0] * quadWeights[trialQuadIndex];
      tempFactor[1] = kernelValue[1] * quadWeights[trialQuadIndex];
      for (j = 0; j < 3; ++j) {
        tempResultFirstComponent[j][0] += tempFactor[0] * trialElementValue[j];
        tempResultFirstComponent[j][1] += tempFactor[1] * trialElementValue[j];
      }
      tempResultSecondComponent[0] += tempFactor[0];
      tempResultSecondComponent[1] += tempFactor[1];
    }

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j) {
        shapeIntegralFirstComponent[i][j][0] +=
            quadWeights[testQuadIndex] *
            dot(testElementValue[i], tempResultFirstComponent[j][0]);
        shapeIntegralFirstComponent[i][j][1] +=
            quadWeights[testQuadIndex] *
            dot(testElementValue[i], tempResultFirstComponent[j][1]);
      }
    shapeIntegralSecondComponent[0] +=
        quadWeights[testQuadIndex] * tempResultSecondComponent[0];
    shapeIntegralSecondComponent[1] +=
        quadWeights[testQuadIndex] * tempResultSecondComponent[1];
  }

  divergenceProduct = M_TWO * M_TWO / testIntElem / trialIntElem;

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegral[i][j][0] =
          -(shiftedWavenumber[0] * shapeIntegralFirstComponent[i][j][0] -
            shiftedWavenumber[1] * shapeIntegralFirstComponent[i][j][1]);
      shapeIntegral[i][j][1] =
          -(shiftedWavenumber[0] * shapeIntegralFirstComponent[i][j][1] +
            shiftedWavenumber[1] * shapeIntegralFirstComponent[i][j][0]);
      shapeIntegral[i][j][0] -=
          divergenceProduct *
          (inverseShiftedWavenumber[0] * shapeIntegralSecondComponent[0] -
           inverseShiftedWavenumber[1] * shapeIntegralSecondComponent[1]);
      shapeIntegral[i][j][1] -=
          divergenceProduct *
          (inverseShiftedWavenumber[0] * shapeIntegralSecondComponent[1] +
           inverseShiftedWavenumber[1] * shapeIntegralSecondComponent[0]);
      shapeIntegral[i][j][0] *= testEdgeLength[i] * trialEdgeLength[j];
      shapeIntegral[i][j][1] *= testEdgeLength[i] * trialEdgeLength[j];
    }

  if (!elementsAreAdjacent(testElement, trialElement, gridsAreDisjoint)) {
    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j) {
        globalRowIndex = myTestLocal2Global[i];
        globalColIndex = myTrialLocal2Global[j];
        globalResult[2 * (globalRowIndex * nTrial + globalColIndex)] +=
            shapeIntegral[i][j][0] * testIntElem * trialIntElem *
            myTestLocalMultipliers[i] * myTrialLocalMultipliers[j];
        globalResult[2 * (globalRowIndex * nTrial + globalColIndex) + 1] +=
            shapeIntegral[i][j][1] * testIntElem * trialIntElem *
            myTestLocalMultipliers[i] * myTrialLocalMultipliers[j];
      }
  }
}
