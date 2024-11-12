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
  REALTYPE testValue[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  REALTYPE trialValue[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                        [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
      shapeIntegral[i][j] = M_ZERO;

  getCorners(grid, elementIndex, corners);
  getJacobian(corners, jacobian);
  getNormalAndIntegrationElement(jacobian, &normal, &intElem);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point = (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    BASIS(TEST, evaluate)(&point, &testValue[0]);
    BASIS(TRIAL, evaluate)(&point, &trialValue[0]);

    for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
        shapeIntegral[i][j] +=
            quadWeights[quadIndex] * testValue[i] * trialValue[j];
  }

  globalIndex = NUMBER_OF_TEST_SHAPE_FUNCTIONS *
                NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * gid;

  for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
      globalResult[globalIndex + i * NUMBER_OF_TRIAL_SHAPE_FUNCTIONS + j] =
          shapeIntegral[i][j] * intElem;
}
