#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_dense_maxwell_multitrace_vector_regular(
    __global uint *testIndices, __global uint *trialIndices,
    __global REALTYPE *testGrid, __global REALTYPE *trialGrid,
    __global uint *testConnectivity, __global uint *trialConnectivity,
    __constant REALTYPE *quadPoints, __constant REALTYPE *quadWeights,
    __global REALTYPE *input, char gridsAreDisjoint,
    __global REALTYPE *globalResult) {
    size_t gid[2] = {get_global_id(0), get_global_id(1)};

    size_t lid = get_local_id(1);
    size_t groupId = get_group_id(1);
    size_t numGroups = get_num_groups(1);

    size_t testIndex = testIndices[gid[0]];
    size_t trialIndex = trialIndices[gid[1]];

    size_t testQuadIndex;
    size_t trialQuadIndex;
    size_t i;
    size_t j;
    size_t k;
    size_t globalRowIndex;
    size_t globalColIndex;
    size_t localIndex;

    REALTYPE3 testGlobalPoint;
    REALTYPE3 trialGlobalPoint;

    REALTYPE3 testCorners[3];
    REALTYPE3 trialCorners[3];

    uint testElement[3];
    uint trialElement[3];

    REALTYPE3 testJac[2];
    REALTYPE3 trialJac[2];

    REALTYPE3 diff;
    REALTYPE dist;
    REALTYPE rdist;

    REALTYPE product[2];

    REALTYPE factor1[2];
    REALTYPE factor2[2];

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
    REALTYPE kernelGradient[3][2];
    REALTYPE tempFactor[2];
    REALTYPE tempResultElectricFirstComponent[3][3][2];
    REALTYPE tempResultElectricSecondComponent[2];
    REALTYPE tempResultMagnetic[3][3][2];
    REALTYPE shapeIntegralElectricFirstComponent[3][3][2];
    REALTYPE shapeIntegralElectricSecondComponent[2];
    REALTYPE shapeIntegralElectric[3][3][2];
    REALTYPE shapeIntegralMagnetic[3][3][2];
    REALTYPE shiftedWavenumber[2] = {M_ZERO, M_ZERO};
    REALTYPE inverseShiftedWavenumber[2] = {M_ZERO, M_ZERO};
    REALTYPE divergenceProduct;

    REALTYPE localCoeffsElectric[3][2];
    REALTYPE localCoeffsMagnetic[3][2];

    __local REALTYPE localResultElectric[WORKGROUP_SIZE][3][3][2];
    __local REALTYPE localResultMagnetic[WORKGROUP_SIZE][3][3][2];
    
#ifdef TRANSMISSION
    REALTYPE kernelValueInt[2];
    REALTYPE kernelGradientInt[3][2];
    REALTYPE tempFactorInt[2];
    REALTYPE tempResultElectricFirstComponentInt[3][3][2];
    REALTYPE tempResultElectricSecondComponentInt[2];
    REALTYPE tempResultMagneticInt[3][3][2];
    REALTYPE shapeIntegralElectricFirstComponentInt[3][3][2];
    REALTYPE shapeIntegralElectricSecondComponentInt[2];
    REALTYPE shapeIntegralElectricInt[3][3][2];
    REALTYPE shapeIntegralMagneticInt[3][3][2];
    REALTYPE shiftedWavenumberInt[2] = {M_ZERO, M_ZERO};
    REALTYPE inverseShiftedWavenumberInt[2] = {M_ZERO, M_ZERO};
    REALTYPE product2[2];

#endif

// Computation of 1i * wavenumber and 1 / (1i * wavenumber)
#ifdef WAVENUMBER_COMPLEX
    shiftedWavenumber[0] = -WAVENUMBER_COMPLEX;
#endif
#ifdef TRANSMISSION
#ifdef WAVENUMBER_INT_COMPLEX
    shiftedWavenumberInt[0] = -WAVENUMBER_INT_COMPLEX;
#endif
#endif


    shiftedWavenumber[1] = WAVENUMBER_REAL;

    inverseShiftedWavenumber[0] = M_ONE /
                                  (shiftedWavenumber[0] * shiftedWavenumber[0] +
                                   shiftedWavenumber[1] * shiftedWavenumber[1]) *
                                  shiftedWavenumber[0];
    inverseShiftedWavenumber[1] = -M_ONE /
                                  (shiftedWavenumber[0] * shiftedWavenumber[0] +
                                   shiftedWavenumber[1] * shiftedWavenumber[1]) *
                                  shiftedWavenumber[1];

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            for (k = 0; k < 2; ++k) {
                shapeIntegralElectricFirstComponent[i][j][k] = M_ZERO;
                shapeIntegralMagnetic[i][j][k] = M_ZERO;
            }
    shapeIntegralElectricSecondComponent[0] = M_ZERO;
    shapeIntegralElectricSecondComponent[1] = M_ZERO;

#ifdef TRANSMISSION
    shiftedWavenumberInt[1] = WAVENUMBER_INT_REAL;

    inverseShiftedWavenumberInt[0] = M_ONE /
                                  (shiftedWavenumberInt[0] * shiftedWavenumberInt[0] +
                                   shiftedWavenumberInt[1] * shiftedWavenumberInt[1]) *
                                  shiftedWavenumberInt[0];
    inverseShiftedWavenumberInt[1] = -M_ONE /
                                  (shiftedWavenumberInt[0] * shiftedWavenumberInt[0] +
                                   shiftedWavenumberInt[1] * shiftedWavenumberInt[1]) *
                                  shiftedWavenumberInt[1];

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            for (k = 0; k < 2; ++k) {
                shapeIntegralElectricFirstComponentInt[i][j][k] = M_ZERO;
                shapeIntegralMagneticInt[i][j][k] = M_ZERO;
            }
    shapeIntegralElectricSecondComponentInt[0] = M_ZERO;
    shapeIntegralElectricSecondComponentInt[1] = M_ZERO;
#endif

    getCorners(testGrid, testIndex, testCorners);
    getCorners(trialGrid, trialIndex, trialCorners);

    getElement(testConnectivity, testIndex, testElement);
    getElement(trialConnectivity, trialIndex, trialElement);

    getJacobian(testCorners, testJac);
    getJacobian(trialCorners, trialJac);

    getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
    getNormalAndIntegrationElement(trialJac, &trialNormal, &trialIntElem);

    computeEdgeLength(testCorners, testEdgeLength);
    computeEdgeLength(trialCorners, trialEdgeLength);

    for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
            ++testQuadIndex) {
        testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex],
                                quadPoints[2 * testQuadIndex + 1]);
        testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
        BASIS(rwg0, evaluate)
        (&testPoint, &testValue[0][0]);
        getPiolaTransform(testIntElem, testJac, testValue, testElementValue);

        for (i = 0; i < 3; ++i)
            for (j = 0; j < 3; ++j) {
                tempResultElectricFirstComponent[i][j][0] = M_ZERO;
                tempResultElectricFirstComponent[i][j][1] = M_ZERO;
                tempResultMagnetic[i][j][0] = M_ZERO;
                tempResultMagnetic[i][j][1] = M_ZERO;

#ifdef TRANSMISSION
                tempResultElectricFirstComponentInt[i][j][0] = M_ZERO;
                tempResultElectricFirstComponentInt[i][j][1] = M_ZERO;
                tempResultMagneticInt[i][j][0] = M_ZERO;
                tempResultMagneticInt[i][j][1] = M_ZERO;
#endif

            }

        tempResultElectricSecondComponent[0] = M_ZERO;
        tempResultElectricSecondComponent[1] = M_ZERO;

#ifdef TRANSMISSION
        tempResultElectricSecondComponentInt[0] = M_ZERO;
        tempResultElectricSecondComponentInt[1] = M_ZERO;
#endif

        for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
                ++trialQuadIndex) {
            trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex],
                                     quadPoints[2 * trialQuadIndex + 1]);
            trialGlobalPoint = getGlobalPoint(trialCorners, &trialPoint);
            BASIS(rwg0, evaluate)
            (&trialPoint, &trialValue[0][0]);
            getPiolaTransform(trialIntElem, trialJac, trialValue, trialElementValue);

            diff = trialGlobalPoint - testGlobalPoint;
            dist = length(diff);
            rdist = M_ONE / dist;

            kernelValue[0] = M_INV_4PI * cos(WAVENUMBER_REAL * dist) * rdist *
                             quadWeights[trialQuadIndex];
            kernelValue[1] = M_INV_4PI * sin(WAVENUMBER_REAL * dist) * rdist *
                             quadWeights[trialQuadIndex];

#ifdef WAVENUMBER_COMPLEX
            kernelValue[0] *= exp(-WAVENUMBER_COMPLEX * dist);
            kernelValue[1] *= exp(-WAVENUMBER_COMPLEX * dist);
#endif

            factor1[0] = kernelValue[0] * rdist * rdist;
            factor1[1] = kernelValue[1] * rdist * rdist;

            factor2[0] = -M_ONE;
            factor2[1] = WAVENUMBER_REAL * dist;

#ifdef WAVENUMBER_COMPLEX
            factor2[0] += -WAVENUMBER_COMPLEX * dist;
#endif

            product[0] = -(factor1[0] * factor2[0] - factor1[1] * factor2[1]);
            product[1] = -(factor1[0] * factor2[1] + factor1[1] * factor2[0]);

            kernelGradient[0][0] = product[0] * diff.x;
            kernelGradient[0][1] = product[1] * diff.x;
            kernelGradient[1][0] = product[0] * diff.y;
            kernelGradient[1][1] = product[1] * diff.y;
            kernelGradient[2][0] = product[0] * diff.z;
            kernelGradient[2][1] = product[1] * diff.z;

            for (j = 0; j < 3; ++j) {
                tempResultElectricFirstComponent[j][0][0] +=
                    kernelValue[0] * trialElementValue[j].x;
                tempResultElectricFirstComponent[j][0][1] +=
                    kernelValue[1] * trialElementValue[j].x;

                tempResultElectricFirstComponent[j][1][0] +=
                    kernelValue[0] * trialElementValue[j].y;
                tempResultElectricFirstComponent[j][1][1] +=
                    kernelValue[1] * trialElementValue[j].y;

                tempResultElectricFirstComponent[j][2][0] +=
                    kernelValue[0] * trialElementValue[j].z;
                tempResultElectricFirstComponent[j][2][1] +=
                    kernelValue[1] * trialElementValue[j].z;

            }

            tempResultElectricSecondComponent[0] += kernelValue[0];
            tempResultElectricSecondComponent[1] += kernelValue[1];

            for (j = 0; j < 3; ++j)
                for (k = 0; k < 2; ++k) {
                    tempResultMagnetic[j][0][k] +=
                        kernelGradient[1][k] * trialElementValue[j].z -
                        kernelGradient[2][k] * trialElementValue[j].y;
                    tempResultMagnetic[j][1][k] +=
                        kernelGradient[2][k] * trialElementValue[j].x -
                        kernelGradient[0][k] * trialElementValue[j].z;
                    tempResultMagnetic[j][2][k] +=
                        kernelGradient[0][k] * trialElementValue[j].y -
                        kernelGradient[1][k] * trialElementValue[j].x;
                }

#ifdef TRANSMISSION
            kernelValueInt[0] = M_INV_4PI * cos(WAVENUMBER_INT_REAL * dist) * rdist *
                             quadWeights[trialQuadIndex];
            kernelValueInt[1] = M_INV_4PI * sin(WAVENUMBER_INT_REAL * dist) * rdist *
                             quadWeights[trialQuadIndex];

#ifdef WAVENUMBER_INT_COMPLEX
            kernelValueInt[0] *= exp(-WAVENUMBER_INT_COMPLEX * dist);
            kernelValueInt[1] *= exp(-WAVENUMBER_INT_COMPLEX * dist);
#endif

            factor1[0] = kernelValueInt[0] * rdist * rdist;
            factor1[1] = kernelValueInt[1] * rdist * rdist;

            factor2[0] = -M_ONE;
            factor2[1] = WAVENUMBER_INT_REAL * dist;

#ifdef WAVENUMBER_INT_COMPLEX
            factor2[0] += -WAVENUMBER_INT_COMPLEX * dist;
#endif

            product[0] = -(factor1[0] * factor2[0] - factor1[1] * factor2[1]);
            product[1] = -(factor1[0] * factor2[1] + factor1[1] * factor2[0]);

            kernelGradientInt[0][0] = product[0] * diff.x;
            kernelGradientInt[0][1] = product[1] * diff.x;
            kernelGradientInt[1][0] = product[0] * diff.y;
            kernelGradientInt[1][1] = product[1] * diff.y;
            kernelGradientInt[2][0] = product[0] * diff.z;
            kernelGradientInt[2][1] = product[1] * diff.z;

            for (j = 0; j < 3; ++j) {
                tempResultElectricFirstComponentInt[j][0][0] +=
                    kernelValueInt[0] * trialElementValue[j].x;
                tempResultElectricFirstComponentInt[j][0][1] +=
                    kernelValueInt[1] * trialElementValue[j].x;

                tempResultElectricFirstComponentInt[j][1][0] +=
                    kernelValueInt[0] * trialElementValue[j].y;
                tempResultElectricFirstComponentInt[j][1][1] +=
                    kernelValueInt[1] * trialElementValue[j].y;

                tempResultElectricFirstComponentInt[j][2][0] +=
                    kernelValueInt[0] * trialElementValue[j].z;
                tempResultElectricFirstComponentInt[j][2][1] +=
                    kernelValueInt[1] * trialElementValue[j].z;

            }

            tempResultElectricSecondComponentInt[0] += kernelValueInt[0];
            tempResultElectricSecondComponentInt[1] += kernelValueInt[1];

            for (j = 0; j < 3; ++j)
                for (k = 0; k < 2; ++k) {
                    tempResultMagneticInt[j][0][k] +=
                        kernelGradientInt[1][k] * trialElementValue[j].z -
                        kernelGradientInt[2][k] * trialElementValue[j].y;
                    tempResultMagneticInt[j][1][k] +=
                        kernelGradientInt[2][k] * trialElementValue[j].x -
                        kernelGradientInt[0][k] * trialElementValue[j].z;
                    tempResultMagneticInt[j][2][k] +=
                        kernelGradientInt[0][k] * trialElementValue[j].y -
                        kernelGradientInt[1][k] * trialElementValue[j].x;
                }
#endif

        }

        for (i = 0; i < 3; ++i)
            for (j = 0; j < 3; ++j) {
                shapeIntegralElectricFirstComponent[i][j][0] +=
                    quadWeights[testQuadIndex] *
                    (testElementValue[i].x * tempResultElectricFirstComponent[j][0][0] +
                     testElementValue[i].y * tempResultElectricFirstComponent[j][1][0] +
                     testElementValue[i].z * tempResultElectricFirstComponent[j][2][0]);
                shapeIntegralElectricFirstComponent[i][j][1] +=
                    quadWeights[testQuadIndex] *
                    (testElementValue[i].x * tempResultElectricFirstComponent[j][0][1] +
                     testElementValue[i].y * tempResultElectricFirstComponent[j][1][1] +
                     testElementValue[i].z * tempResultElectricFirstComponent[j][2][1]);
            }

        shapeIntegralElectricSecondComponent[0] +=
            quadWeights[testQuadIndex] * tempResultElectricSecondComponent[0];
        shapeIntegralElectricSecondComponent[1] +=
            quadWeights[testQuadIndex] * tempResultElectricSecondComponent[1];

        for (i = 0; i < 3; ++i)
            for (j = 0; j < 3; ++j)
                for (k = 0; k < 2; ++k)
                    shapeIntegralMagnetic[i][j][k] -=
                        quadWeights[testQuadIndex] *
                        (testElementValue[i].x * tempResultMagnetic[j][0][k] +
                         testElementValue[i].y * tempResultMagnetic[j][1][k] +
                         testElementValue[i].z * tempResultMagnetic[j][2][k]);

#ifdef TRANSMISSION
        for (i = 0; i < 3; ++i)
            for (j = 0; j < 3; ++j) {
                shapeIntegralElectricFirstComponentInt[i][j][0] +=
                    quadWeights[testQuadIndex] *
                    (testElementValue[i].x * tempResultElectricFirstComponentInt[j][0][0] +
                     testElementValue[i].y * tempResultElectricFirstComponentInt[j][1][0] +
                     testElementValue[i].z * tempResultElectricFirstComponentInt[j][2][0]);
                shapeIntegralElectricFirstComponentInt[i][j][1] +=
                    quadWeights[testQuadIndex] *
                    (testElementValue[i].x * tempResultElectricFirstComponentInt[j][0][1] +
                     testElementValue[i].y * tempResultElectricFirstComponentInt[j][1][1] +
                     testElementValue[i].z * tempResultElectricFirstComponentInt[j][2][1]);
            }

        shapeIntegralElectricSecondComponentInt[0] +=
            quadWeights[testQuadIndex] * tempResultElectricSecondComponentInt[0];
        shapeIntegralElectricSecondComponentInt[1] +=
            quadWeights[testQuadIndex] * tempResultElectricSecondComponentInt[1];

        for (i = 0; i < 3; ++i)
            for (j = 0; j < 3; ++j)
                for (k = 0; k < 2; ++k)
                    shapeIntegralMagneticInt[i][j][k] -=
                        quadWeights[testQuadIndex] *
                        (testElementValue[i].x * tempResultMagneticInt[j][0][k] +
                         testElementValue[i].y * tempResultMagneticInt[j][1][k] +
                         testElementValue[i].z * tempResultMagneticInt[j][2][k]);
#endif

    }

    divergenceProduct = M_TWO * M_TWO / testIntElem / trialIntElem;

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) {
            shapeIntegralElectric[i][j][0] = -(
                                                 shiftedWavenumber[0] * shapeIntegralElectricFirstComponent[i][j][0] -
                                                 shiftedWavenumber[1] * shapeIntegralElectricFirstComponent[i][j][1]);
            shapeIntegralElectric[i][j][1] = -(
                                                 shiftedWavenumber[0] * shapeIntegralElectricFirstComponent[i][j][1] +
                                                 shiftedWavenumber[1] * shapeIntegralElectricFirstComponent[i][j][0]);
            shapeIntegralElectric[i][j][0] -=
                divergenceProduct * (inverseShiftedWavenumber[0] *
                                     shapeIntegralElectricSecondComponent[0] -
                                     inverseShiftedWavenumber[1] *
                                     shapeIntegralElectricSecondComponent[1]);
            shapeIntegralElectric[i][j][1] -=
                divergenceProduct * (inverseShiftedWavenumber[0] *
                                     shapeIntegralElectricSecondComponent[1] +
                                     inverseShiftedWavenumber[1] *
                                     shapeIntegralElectricSecondComponent[0]);
            shapeIntegralElectric[i][j][0] *= testEdgeLength[i] * trialEdgeLength[j];
            shapeIntegralElectric[i][j][1] *= testEdgeLength[i] * trialEdgeLength[j];
        }

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            for (k = 0; k < 2; ++k)
                shapeIntegralMagnetic[i][j][k] *=
                    testEdgeLength[i] * trialEdgeLength[j];

#ifdef TRANSMISSION
    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) {
            shapeIntegralElectricInt[i][j][0] = -(
                                                 shiftedWavenumberInt[0] * shapeIntegralElectricFirstComponentInt[i][j][0] -
                                                 shiftedWavenumberInt[1] * shapeIntegralElectricFirstComponentInt[i][j][1]);
            shapeIntegralElectricInt[i][j][1] = -(
                                                 shiftedWavenumberInt[0] * shapeIntegralElectricFirstComponentInt[i][j][1] +
                                                 shiftedWavenumberInt[1] * shapeIntegralElectricFirstComponentInt[i][j][0]);
            shapeIntegralElectricInt[i][j][0] -=
                divergenceProduct * (inverseShiftedWavenumberInt[0] *
                                     shapeIntegralElectricSecondComponentInt[0] -
                                     inverseShiftedWavenumberInt[1] *
                                     shapeIntegralElectricSecondComponentInt[1]);
            shapeIntegralElectricInt[i][j][1] -=
                divergenceProduct * (inverseShiftedWavenumberInt[0] *
                                     shapeIntegralElectricSecondComponentInt[1] +
                                     inverseShiftedWavenumberInt[1] *
                                     shapeIntegralElectricSecondComponentInt[0]);
            shapeIntegralElectricInt[i][j][0] *= testEdgeLength[i] * trialEdgeLength[j];
            shapeIntegralElectricInt[i][j][1] *= testEdgeLength[i] * trialEdgeLength[j];
        }

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            for (k = 0; k < 2; ++k)
                shapeIntegralMagneticInt[i][j][k] *=
                    testEdgeLength[i] * trialEdgeLength[j];
#endif


    if (!elementsAreAdjacent(testElement, trialElement, gridsAreDisjoint)) {
        for (j = 0; j < 3; ++j) {
            localCoeffsElectric[j][0] = input[2 * (3 * trialIndex + j)];
            localCoeffsElectric[j][1] = input[2 * (3 * trialIndex + j) + 1];

            localCoeffsMagnetic[j][0] =
                input[6 * TRIAL0_NUMBER_OF_ELEMENTS + 2 * (3 * trialIndex + j)];
            localCoeffsMagnetic[j][1] =
                input[6 * TRIAL0_NUMBER_OF_ELEMENTS + 2 * (3 * trialIndex + j) + 1];
        }

        for (i = 0; i < 3; ++i)
            for (j = 0; j < 3; ++j) {
                localResultElectric[lid][i][j][0] =
                    testIntElem * trialIntElem *
                    ((shapeIntegralMagnetic[i][j][0] * localCoeffsElectric[j][0] -
                      shapeIntegralMagnetic[i][j][1] * localCoeffsElectric[j][1]) +
                     (shapeIntegralElectric[i][j][0] * localCoeffsMagnetic[j][0] -
                      shapeIntegralElectric[i][j][1] * localCoeffsMagnetic[j][1]));
                localResultElectric[lid][i][j][1] =
                    testIntElem * trialIntElem *
                    ((shapeIntegralMagnetic[i][j][0] * localCoeffsElectric[j][1] +
                      shapeIntegralMagnetic[i][j][1] * localCoeffsElectric[j][0]) +
                     (shapeIntegralElectric[i][j][0] * localCoeffsMagnetic[j][1] +
                      shapeIntegralElectric[i][j][1] * localCoeffsMagnetic[j][0]));

                localResultMagnetic[lid][i][j][0] =
                    testIntElem * trialIntElem *
                    (-(shapeIntegralElectric[i][j][0] * localCoeffsElectric[j][0] -
                       shapeIntegralElectric[i][j][1] * localCoeffsElectric[j][1]) +
                     (shapeIntegralMagnetic[i][j][0] * localCoeffsMagnetic[j][0] -
                      shapeIntegralMagnetic[i][j][1] * localCoeffsMagnetic[j][1]));
                localResultMagnetic[lid][i][j][1] =
                    testIntElem * trialIntElem *
                    (-(shapeIntegralElectric[i][j][0] * localCoeffsElectric[j][1] +
                       shapeIntegralElectric[i][j][1] * localCoeffsElectric[j][0]) +
                     (shapeIntegralMagnetic[i][j][0] * localCoeffsMagnetic[j][1] +
                      shapeIntegralMagnetic[i][j][1] * localCoeffsMagnetic[j][0]));

#ifdef TRANSMISSION
                factor1[0] = SQRT_MU_EPS_RATIO_REAL;
                factor2[0] = SQRT_EPS_MU_RATIO_REAL;

                factor1[1] = M_ZERO;
                factor2[1] = M_ZERO;
#ifdef SQRT_MU_EPS_RATIO_COMPLEX
                factor1[1] = SQRT_MU_EPS_RATIO_COMPLEX;
#endif
#ifdef SQRT_EPS_MU_RATIO_COMPLEX
                factor2[1] = SQRT_EPS_MU_RATIO_COMPLEX;
#endif
                product[0] = CMP_MULT_REAL(shapeIntegralElectricInt[i][j], factor1);
                product[1] = CMP_MULT_IMAG(shapeIntegralElectricInt[i][j], factor1);
                product2[0] = CMP_MULT_REAL(shapeIntegralElectricInt[i][j], factor2);
                product2[1] = CMP_MULT_IMAG(shapeIntegralElectricInt[i][j], factor2);

                localResultElectric[lid][i][j][0] +=
                    testIntElem * trialIntElem *
                    ((shapeIntegralMagneticInt[i][j][0] * localCoeffsElectric[j][0] -
                      shapeIntegralMagneticInt[i][j][1] * localCoeffsElectric[j][1]) +
                     (product[0] * localCoeffsMagnetic[j][0] -
                      product[1] * localCoeffsMagnetic[j][1]));
                localResultElectric[lid][i][j][1] +=
                    testIntElem * trialIntElem *
                    ((shapeIntegralMagneticInt[i][j][0] * localCoeffsElectric[j][1] +
                      shapeIntegralMagneticInt[i][j][1] * localCoeffsElectric[j][0]) +
                     (product[0] * localCoeffsMagnetic[j][1] +
                      product[1] * localCoeffsMagnetic[j][0]));

                localResultMagnetic[lid][i][j][0] +=
                    testIntElem * trialIntElem *
                    (-(product2[0] * localCoeffsElectric[j][0] -
                       product2[1] * localCoeffsElectric[j][1]) +
                     (shapeIntegralMagneticInt[i][j][0] * localCoeffsMagnetic[j][0] -
                      shapeIntegralMagneticInt[i][j][1] * localCoeffsMagnetic[j][1]));
                localResultMagnetic[lid][i][j][1] +=
                    testIntElem * trialIntElem *
                    (-(product2[0] * localCoeffsElectric[j][1] +
                       product2[1] * localCoeffsElectric[j][0]) +
                     (shapeIntegralMagneticInt[i][j][0] * localCoeffsMagnetic[j][1] +
                      shapeIntegralMagneticInt[i][j][1] * localCoeffsMagnetic[j][0]));

#endif

            }



    } else {
        for (i = 0; i < 3; ++i)
            for (j = 0; j < 3; ++j) {
                localResultElectric[lid][i][j][0] = M_ZERO;
                localResultElectric[lid][i][j][1] = M_ZERO;

                localResultMagnetic[lid][i][j][0] = M_ZERO;
                localResultMagnetic[lid][i][j][1] = M_ZERO;
            }
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        for (localIndex = 1; localIndex < WORKGROUP_SIZE; ++localIndex)
            for (i = 0; i < 3; ++i)
                for (j = 0; j < 3; ++j) {
                    localResultElectric[0][i][j][0] +=
                        localResultElectric[localIndex][i][j][0];
                    localResultElectric[0][i][j][1] +=
                        localResultElectric[localIndex][i][j][1];

                    localResultMagnetic[0][i][j][0] +=
                        localResultMagnetic[localIndex][i][j][0];
                    localResultMagnetic[0][i][j][1] +=
                        localResultMagnetic[localIndex][i][j][1];
                }

        for (i = 0; i < 3; ++i) {
            for (j = 1; j < 3; ++j) {
                localResultElectric[0][i][0][0] += localResultElectric[0][i][j][0];
                localResultElectric[0][i][0][1] += localResultElectric[0][i][j][1];

                localResultMagnetic[0][i][0][0] += localResultMagnetic[0][i][j][0];
                localResultMagnetic[0][i][0][1] += localResultMagnetic[0][i][j][1];
            }

            globalResult[2 * (numGroups * (3 * testIndex + i) + groupId)] +=
                localResultElectric[0][i][0][0];
            globalResult[2 * (numGroups * (3 * testIndex + i) + groupId) + 1] +=
                localResultElectric[0][i][0][1];

            globalResult[6 * TRIAL0_NUMBER_OF_ELEMENTS * numGroups +
                           2 * (numGroups * (3 * testIndex + i) + groupId)] +=
                             localResultMagnetic[0][i][0][0];
            globalResult[6 * TRIAL0_NUMBER_OF_ELEMENTS * numGroups +
                           2 * (numGroups * (3 * testIndex + i) + groupId) + 1] +=
                             localResultMagnetic[0][i][0][1];
        }
    }
}
