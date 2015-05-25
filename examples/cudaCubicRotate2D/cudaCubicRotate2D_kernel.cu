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
#include <cutil.h>
#include <memcpy.cu>
#include <cubicPrefilter2D.cu>
#include <cubicTex2D.cu>

#define PI ((double)3.14159265358979323846264338327950288419716939937510)
texture<float, 2, cudaReadModeElementType> coeffs;  //2D texture


__global__ void
interpolate_kernel(float* output, uint width, float2 extent, float2 a, float2 shift, bool masking)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint i = __umul24(y, width) + x;

	float x0 = (float)x;
	float y0 = (float)y;
	float x1 = a.x * x0 - a.y * y0 + shift.x;
	float y1 = a.x * y0 + a.y * x0 + shift.y;

	bool inside =
		-0.5f < x1 && x1 < (extent.x - 0.5f) &&
		-0.5f < y1 && y1 < (extent.y - 0.5f);

	if (masking && !inside)
	{
		output[i] = 0.0f;
	}
	else
	{
		output[i] = cubicTex2D(coeffs, x1, y1);
	}
}


extern "C" cudaPitchedPtr interpolate(
	uint width, uint height, double angle,
	double xShift, double yShift, double xOrigin, double yOrigin, int masking)
{
	// Prepare the geometry
	angle *= PI / 180.0;
	float2 a = make_float2((float)cos(angle), (float)sin(angle));
	double x0 = a.x * (xShift + xOrigin) - a.y * (yShift + yOrigin);
	double y0 = a.y * (xShift + xOrigin) + a.x * (yShift + yOrigin);
	xShift = xOrigin - x0;
	yShift = yOrigin - y0;

	// Allocate the output image
	float* output;
	CUDA_SAFE_CALL(cudaMalloc((void**)&output, width * height * sizeof(float)));

	// Visit all pixels of the output image and assign their value
	dim3 blockSize(min(PowTwoDivider(width), 16), min(PowTwoDivider(height), 16));
	dim3 gridSize(width / blockSize.x, height / blockSize.y);
	float2 shift = make_float2((float)xShift, (float)yShift);
	float2 extent = make_float2((float)width, (float)height);
	interpolate_kernel<<<gridSize, blockSize>>>(output, width, extent, a, shift, masking != 0);
	CUT_CHECK_ERROR("kernel failed");

	return make_cudaPitchedPtr(output, width * sizeof(float), width, height);
}


extern "C" void initTexture(cudaPitchedPtr bsplineCoeffs, uint width, uint height)
{
	// Create the B-spline coefficients texture
	cudaChannelFormatDesc channelDescCoeff = cudaCreateChannelDesc<float>();
	cudaArray *coeffArray = 0;
	CUDA_SAFE_CALL(cudaMallocArray(&coeffArray, &channelDescCoeff, width, height));
	CUDA_SAFE_CALL(cudaMemcpy2DToArray(coeffArray, 0, 0, bsplineCoeffs.ptr, bsplineCoeffs.pitch, width * sizeof(float), height, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaBindTextureToArray(coeffs, coeffArray, channelDescCoeff));
	coeffs.normalized = false;  // access with normalized texture coordinates
	coeffs.filterMode = cudaFilterModeLinear;
}


extern "C" void MyCubicBSplinePrefilter2DTimer(cudaPitchedPtr image, uint width, uint height)
{
	return CubicBSplinePrefilter2DTimer((float*)image.ptr, (uint)image.pitch, width, height);
}
