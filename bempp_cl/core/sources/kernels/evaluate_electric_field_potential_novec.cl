#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void kernel_function(
    __global REALTYPE *grid, __global uint *indices, __global int *normalSigns,
    __global REALTYPE *evalPoints, __global REALTYPE *coefficients,
    __constant REALTYPE *quadPoints, __constant REALTYPE *quadWeights,
    __global REALTYPE *globalResult, __global REALTYPE *kernel_parameters) {
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
  REALTYPE3 diff;

  REALTYPE dist;

  REALTYPE2 point;

  REALTYPE intElem;
  REALTYPE twiceInvIntElem;

  size_t quadIndex;
  size_t i, j;

  REALTYPE shapeIntegral[3][3][2];
  REALTYPE shiftedWavenumber[2] = {M_ZERO, M_ZERO};
  REALTYPE inverseShiftedWavenumber[2] = {M_ZERO, M_ZERO};

  __local REALTYPE localResult[WORKGROUP_SIZE][3][2];
  REALTYPE kernelValue[2];
  REALTYPE gradKernelValue[3][2];

  REALTYPE tempResult[3][3][2];
  REALTYPE myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS][2];

  REALTYPE product[2];
  REALTYPE factor1[2];
  REALTYPE factor2[2];

  REALTYPE edgeLengths[3];

  REALTYPE3 evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);

  for (i = 0; i < 3; ++i) {
    myCoefficients[i][0] = coefficients[2 * (3 * elementIndex + i)];
    myCoefficients[i][1] = coefficients[2 * (3 * elementIndex + i) + 1];
  }

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
      shapeIntegral[i][j][0] = M_ZERO;
      shapeIntegral[i][j][1] = M_ZERO;
    }

  getCorners(grid, elementIndex, corners);
  getJacobian(corners, jacobian);
  getNormalAndIntegrationElement(jacobian, &normal, &intElem);

  updateNormals(elementIndex, normalSigns, &normal);

  computeEdgeLength(corners, edgeLengths);

  twiceInvIntElem = M_TWO / intElem;

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point =
        (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    surfaceGlobalPoint = getGlobalPoint(corners, &point);
    BASIS(SHAPESET, evaluate)(&point, &basisValue[0][0]);
    getPiolaTransform(intElem, jacobian, basisValue, elementValue);

    dist = distance(evalGlobalPoint, surfaceGlobalPoint);
    diff = evalGlobalPoint - surfaceGlobalPoint;

    kernelValue[0] = M_INV_4PI * cos(kernel_parameters[0] * dist) / dist;
    kernelValue[1] = M_INV_4PI * sin(kernel_parameters[0] * dist) / dist;

    if (kernel_parameters[1] != M_ZERO){
      kernelValue[0] *= exp(-kernel_parameters[1] * dist);
      kernelValue[1] *= exp(-kernel_parameters[1] * dist);
    }

    factor1[0] = kernelValue[0] / (dist * dist);
    factor1[1] = kernelValue[1] / (dist * dist);

    factor2[0] = -M_ONE;
    factor2[1] = kernel_parameters[0] * dist;

    if (kernel_parameters[1] != M_ZERO)
      factor2[0] += -kernel_parameters[1] * dist;


    product[0] = (factor1[0] * factor2[0] - factor1[1] * factor2[1]);
    product[1] = (factor1[0] * factor2[1] + factor1[1] * factor2[0]);

    gradKernelValue[0][0] = product[0] * diff.x;
    gradKernelValue[0][1] = product[1] * diff.x;
    gradKernelValue[1][0] = product[0] * diff.y;
    gradKernelValue[1][1] = product[1] * diff.y;
    gradKernelValue[2][0] = product[0] * diff.z;
    gradKernelValue[2][1] = product[1] * diff.z;

    factor1[0] = CMP_MULT_REAL(shiftedWavenumber, kernelValue);
    factor1[1] = CMP_MULT_IMAG(shiftedWavenumber, kernelValue);

    for (i = 0; i < 3; ++i) {
      tempResult[i][0][0] = factor1[0] * elementValue[i].x;
      tempResult[i][0][1] = factor1[1] * elementValue[i].x;
      tempResult[i][1][0] = factor1[0] * elementValue[i].y;
      tempResult[i][1][1] = factor1[1] * elementValue[i].y;
      tempResult[i][2][0] = factor1[0] * elementValue[i].z;
      tempResult[i][2][1] = factor1[1] * elementValue[i].z;
    }

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j) {
        tempResult[i][j][0] -=
            CMP_MULT_REAL(inverseShiftedWavenumber, gradKernelValue[j]) *
            twiceInvIntElem;
        tempResult[i][j][1] -=
            CMP_MULT_IMAG(inverseShiftedWavenumber, gradKernelValue[j]) *
            twiceInvIntElem;
      }

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j) {
        shapeIntegral[i][j][0] += tempResult[i][j][0] * quadWeights[quadIndex];
        shapeIntegral[i][j][1] += tempResult[i][j][1] * quadWeights[quadIndex];
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
