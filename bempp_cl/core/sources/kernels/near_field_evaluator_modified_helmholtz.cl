#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPEVEC))) void kernel_function(
	 __global REALTYPE* grid,
	 __global int* neighborIndices,
	 __global int* neighborIndexptr,
	 __global REALTYPE* localPoints,
	 __global REALTYPE* coefficients,
	 __global REALTYPE* result,
	 __global REALTYPE* kernelParameters,
	 uint nelements
	 )
{
  size_t gid = get_global_id(0);

  int indexStart = neighborIndexptr[gid];
  int indexEnd = neighborIndexptr[1 + gid];

  int nIndices = indexEnd - indexStart;
  int nChunks = (NPOINTS * nIndices) / VEC_LENGTH;

  REALTYPE globalSourcePoints[3 * MAX_POINTS];
  REALTYPE localCoefficients[MAX_POINTS];
  REALTYPE3 targetCorners[3];
  REALTYPE3 sourceCorners[3];
  REALTYPE3 globalPoint;
  REALTYPEVEC sourceVecPoint[3];
  REALTYPEVEC diffVec[3];
  REALTYPE diff[3];
  REALTYPEVEC distVec;
  REALTYPEVEC rdistVec;
  REALTYPEVEC expvalVec;
  REALTYPEVEC tmpVec;
  REALTYPE dist;
  REALTYPE tmp;
  REALTYPE rdist;
  REALTYPE expval;
  REALTYPE3 targetPoint;
  REALTYPE2 point;
  REALTYPEVEC coeffsVec;
  REALTYPEVEC resultVec[4];
  REALTYPE resultSingle[4];
  
  getCorners(grid, gid, targetCorners);

  // First compute global points

  int count = 0;
  for (int i = 0; i < nIndices; ++i){
      int elem = neighborIndices[i + indexStart];
      getCorners(grid, elem, sourceCorners);
      for (int localIndex = 0; localIndex < NPOINTS; ++localIndex)
      	  {
		point = (REALTYPE2)(localPoints[2 * localIndex],
		                    localPoints[2 * localIndex + 1]);
		globalPoint = getGlobalPoint(sourceCorners, &point);
		globalSourcePoints[3 * count] = globalPoint.x;
		globalSourcePoints[3 * count + 1] = globalPoint.y;
		globalSourcePoints[3 * count + 2] = globalPoint.z;
		localCoefficients[count] = coefficients[NPOINTS * elem + localIndex];
		count += 1;
	}
   }


  for (int targetIndex = 0; targetIndex < NPOINTS; targetIndex++)
    {

      for (int i = 0; i < 4; i++){
	resultVec[i] = M_ZERO;
	resultSingle[i] = M_ZERO;
      }

      point = (REALTYPE2)(localPoints[2 * targetIndex],
      		    localPoints[2 * targetIndex + 1]);

      targetPoint = getGlobalPoint(targetCorners, &point);

      for (int chunkIndex = 0; chunkIndex < nChunks; chunkIndex++){
	// Fill chunk
	for (int vecIndex = 0; vecIndex < VEC_LENGTH; vecIndex++){
	  VEC_ELEMENT(sourceVecPoint[0], vecIndex) = globalSourcePoints[3 * (VEC_LENGTH * chunkIndex + vecIndex) + 0];
	  VEC_ELEMENT(sourceVecPoint[1], vecIndex) = globalSourcePoints[3 * (VEC_LENGTH * chunkIndex + vecIndex) + 1];
	  VEC_ELEMENT(sourceVecPoint[2], vecIndex) = globalSourcePoints[3 * (VEC_LENGTH * chunkIndex + vecIndex) + 2];
	  VEC_ELEMENT(coeffsVec,vecIndex) = localCoefficients[VEC_LENGTH * chunkIndex + vecIndex];
	}

	diffVec[0] = targetPoint.x - sourceVecPoint[0];
	diffVec[1] = targetPoint.y - sourceVecPoint[1];
	diffVec[2] = targetPoint.z - sourceVecPoint[2];

	
	distVec = sqrt(diffVec[0] * diffVec[0] + diffVec[1] * diffVec[1] + diffVec[2] * diffVec[2]);
	rdistVec = M_ONE / distVec;
	// Check for zero dist case
	for (int vecIndex = 0; vecIndex < VEC_LENGTH; vecIndex++){
	  if ((VEC_ELEMENT(diffVec[0], vecIndex) == M_ZERO) && (VEC_ELEMENT(diffVec[1], vecIndex) == M_ZERO) && (VEC_ELEMENT(diffVec[2], vecIndex) == M_ZERO))
	    VEC_ELEMENT(rdistVec, vecIndex) = M_ZERO;
	}

        expvalVec = exp(-kernelParameters[0] * distVec) * rdistVec * coeffsVec * M_INV_4PI;
	resultVec[0] += expvalVec;
	resultVec[1] += (-kernelParameters[0] * distVec - M_ONE) * expvalVec * rdistVec * rdistVec * diffVec[0];
	resultVec[2] += (-kernelParameters[0] * distVec - M_ONE) * expvalVec * rdistVec * rdistVec * diffVec[1];
	resultVec[3] += (-kernelParameters[0] * distVec - M_ONE) * expvalVec * rdistVec * rdistVec * diffVec[2];
	
	
      }

      // Now process the remainder scalar points
      for (int remainderIndex = nChunks * VEC_LENGTH; remainderIndex < NPOINTS * nIndices; remainderIndex++)
	{
	  diff[0] = targetPoint.x - globalSourcePoints[3 * remainderIndex + 0];
	  diff[1] = targetPoint.y - globalSourcePoints[3 * remainderIndex + 1];
	  diff[2] = targetPoint.z - globalSourcePoints[3 * remainderIndex + 2];

	  dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
	  rdist = M_ONE / dist;
	  if ((diff[0] == M_ZERO) && (diff[1] == M_ZERO) && (diff[2] == M_ZERO))
	    rdist = M_ZERO;

	  expval = exp(-kernelParameters[0] * dist) * rdist * localCoefficients[remainderIndex] * M_INV_4PI;
	  resultSingle[0] += expval;
	  resultSingle[1] += (-kernelParameters[0] * dist - M_ONE) * expval * rdist * rdist * diff[0];
	  resultSingle[2] += (-kernelParameters[0] * dist - M_ONE) * expval * rdist * rdist * diff[1];
	  resultSingle[3] += (-kernelParameters[0] * dist - M_ONE) * expval * rdist * rdist * diff[2];
	}
      
      result[4 * (NPOINTS * gid + targetIndex) + 0] = resultSingle[0];
      result[4 * (NPOINTS * gid + targetIndex) + 1] = resultSingle[1];
      result[4 * (NPOINTS * gid + targetIndex) + 2] = resultSingle[2];
      result[4 * (NPOINTS * gid + targetIndex) + 3] = resultSingle[3];
      
      
      for (int vecIndex = 0; vecIndex < VEC_LENGTH; vecIndex++){
	result[4 * (NPOINTS * gid + targetIndex) + 0] += VEC_ELEMENT(resultVec[0], vecIndex);
	result[4 * (NPOINTS * gid + targetIndex) + 1] += VEC_ELEMENT(resultVec[1], vecIndex);
	result[4 * (NPOINTS * gid + targetIndex) + 2] += VEC_ELEMENT(resultVec[2], vecIndex);
	result[4 * (NPOINTS * gid + targetIndex) + 3] += VEC_ELEMENT(resultVec[3], vecIndex);
      }
      
    }

}      
      


