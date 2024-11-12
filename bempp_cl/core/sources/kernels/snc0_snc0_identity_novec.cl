#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"

__kernel void evaluate(__global REALTYPE *grid,
                       __global uint *indices,
                       __global int *testNormalSigns, __global int *trialNormalSigns,
                       __constant REALTYPE* quadPoints,
                       __constant REALTYPE *quadWeights,
                       __global REALTYPE *globalResult, int nelements) {
  /* Variable declarations */

  // globalId(0) is always zero
  size_t gid = get_global_id(1);
  size_t elementIndex = indices[gid];

  size_t quadIndex;
  size_t globalIndex;
  size_t i, j;

  REALTYPE2 point;
  REALTYPE3 globalPoint;
  REALTYPE3 corners[3];
  REALTYPE3 jacobian[2];
  REALTYPE intElem;
  REALTYPE3 normal;
  REALTYPE basisValue[3][2];
  REALTYPE3 elementValue[3];
  REALTYPE3 crossValue1;
  REALTYPE3 crossValue2;
  REALTYPE edgeLength[3];

  REALTYPE shapeIntegral[3]
                        [3];

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      shapeIntegral[i][j] = M_ZERO;

  getCorners(grid, elementIndex, corners);
  getJacobian(corners, jacobian);
  getNormalAndIntegrationElement(jacobian, &normal, &intElem);
  computeEdgeLength(corners, edgeLength);

  updateNormals(elementIndex, testNormalSigns, &normal);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point = (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    globalPoint = getGlobalPoint(corners, &point);
    BASIS(TEST, evaluate)(&point, &basisValue[0][0]);
    getPiolaTransform(intElem, jacobian, basisValue, elementValue);

    for (i = 0; i < 3; ++i){
      crossValue1 = cross(normal, elementValue[i]);
      for (j = 0; j < 3; ++j){
        crossValue2 = cross(normal, elementValue[j]);
        shapeIntegral[i][j] +=
            quadWeights[quadIndex] * 
              dot(crossValue2, crossValue1) * edgeLength[i] * edgeLength[j];
      }
    }
  }

  globalIndex = 9 * gid; 

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      globalResult[globalIndex + i * 3 + j] =
          shapeIntegral[i][j] * intElem;
}
