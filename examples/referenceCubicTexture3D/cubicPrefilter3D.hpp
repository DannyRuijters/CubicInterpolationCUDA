/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2010, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.

When using this code in a scientific project, please cite one or all of the
following papers:
*  Daniel Ruijters and Philippe Thévenaz,
   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
   The Computer Journal, vol. 55, no. 1, pp. 15-20, January 2012.
   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
*  Daniel Ruijters, Bart M. ter Haar Romeny, and Paul Suetens,
   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
   Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.
\*--------------------------------------------------------------------------*/

#include <stdio.h>
#include <math.h>
#include "internal/cubicPrefilter_kernel.cu"

//--------------------------------------------------------------------------
// Global CUDA procedures
//--------------------------------------------------------------------------
void SamplesToCoefficients3DX(
	uint y, uint z,
	float* volume,		// in-place processing
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in x-direction
	const uint startIdx = (z * height + y) * width;
	ConvertToInterpolationCoefficients(volume + startIdx, width, sizeof(float));
}

void SamplesToCoefficients3DY(
	uint x, uint z,
	float* volume,		// in-place processing
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in y-direction
	const uint startIdx = z * height * width + x;
	ConvertToInterpolationCoefficients(volume + startIdx, height, width * sizeof(float));
}

void SamplesToCoefficients3DZ(
	uint x, uint y,
	float* volume,		// in-place processing
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in z-direction
	const uint startIdx = y * width + x;
	const uint slice = height * width;
	ConvertToInterpolationCoefficients(volume + startIdx, depth, slice * sizeof(float));
}

//--------------------------------------------------------------------------
// Exported functions
//--------------------------------------------------------------------------

//! Convert the voxel values into cubic b-spline coefficients
//! @param volume  pointer to the voxel volume in GPU (device) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
//! @note Prints stopwatch feedback
extern "C"
void CubicBSplinePrefilter3DTimer(float* volume, uint width, uint height, uint depth)
{
	printf("\nCubic B-Spline Prefilter timer:\n");
	uint hTimer;
	CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	CUT_SAFE_CALL(cutResetTimer(hTimer));
    CUT_SAFE_CALL(cutStartTimer(hTimer));

	// Replace the voxel values by the b-spline coefficients
	for (uint z = 0; z < depth; z++)
	for (uint y = 0; y < height; y++)
	{
		SamplesToCoefficients3DX(y, z, volume, width, height, depth);
	}

	CUT_SAFE_CALL(cutStopTimer(hTimer));
    double timerValueX = cutGetTimerValue(hTimer);
    printf("x-direction : %f msec\n", timerValueX);
	CUT_SAFE_CALL(cutResetTimer(hTimer));
    CUT_SAFE_CALL(cutStartTimer(hTimer));

	for (uint z = 0; z < depth; z++)
	for (uint x = 0; x < width; x++)
	{
		SamplesToCoefficients3DY(x, z, volume, width, height, depth);
	}

	CUT_SAFE_CALL(cutStopTimer(hTimer));
    double timerValueY = cutGetTimerValue(hTimer);
    printf("y-direction : %f msec\n", timerValueY);
	CUT_SAFE_CALL(cutResetTimer(hTimer));
    CUT_SAFE_CALL(cutStartTimer(hTimer));

	for (uint y = 0; y < height; y++)
	for (uint x = 0; x < width; x++)
	{
		SamplesToCoefficients3DZ(x, y, volume, width, height, depth);
	}

	CUT_SAFE_CALL(cutStopTimer(hTimer));
    double timerValueZ = cutGetTimerValue(hTimer);
    printf("z-direction : %f msec\n", timerValueZ);
	printf("total : %f msec\n\n", timerValueX+timerValueY+timerValueZ);
}

