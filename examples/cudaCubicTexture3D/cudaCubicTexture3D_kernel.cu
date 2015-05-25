/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2009, Danny Ruijters. All rights reserved.
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
#include <cubicPrefilter3D.cu>
#include <cubicTex3D.cu>

texture<uchar, 3, cudaReadModeNormalizedFloat> tex;  //3D texture
texture<float, 3, cudaReadModeElementType> coeffs;  //3D texture


__global__ void
render_kernel(uchar* output, uint2 imageExtent, float3 volumeExtent, float w, uint filterMethod)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	float u = x / (float)imageExtent.x;
	float v = y / (float)imageExtent.y;
	float3 coord = volumeExtent * make_float3(u, v, w);

	// read from 3D texture
	float voxel;
	switch (filterMethod)
	{
		case 0:  //nearest neighbor
		case 1: voxel = linearTex3D(tex, coord); break;  //linear
		case 2: voxel = cubicTex3DSimple(coeffs, coord); break;  //simple cubic
		case 3: voxel = cubicTex3D(coeffs, coord); break;  //fast cubic
		case 4: voxel = cubicTex3D(tex, coord); break;  //non-prefiltered, fast cubic
	}

	// write output color
	uint i = __umul24(y, imageExtent.x) + x;
	output[i] = __saturatef(voxel) * 255;
}


// render image using CUDA
extern "C" void render(uchar* output, uint2 imageExtent, uint3 volumeSize, float w, uint filterMethod)
{
	// set texture parameters
	tex.filterMode = (filterMethod == 0) ? cudaFilterModePoint : cudaFilterModeLinear;

	// call CUDA kernel, writing results to PBO
	const dim3 blockSize(min(PowTwoDivider(imageExtent.x), 16), min(PowTwoDivider(imageExtent.y), 16));
	const dim3 gridSize(imageExtent.x / blockSize.x, imageExtent.y / blockSize.y);
	const float3 volumeExtent = make_float3((float)volumeSize.x, (float)volumeSize.y, (float)volumeSize.z);
	render_kernel<<<gridSize, blockSize>>>(output, imageExtent, volumeExtent, w, filterMethod);
	CUT_CHECK_ERROR("kernel failed");
}


// intialize the textures, and calculate the cubic B-spline coefficients
extern "C" void initCuda(const uchar* voxels, uint3 volumeSize)
{
	// calculate the b-spline coefficients
	cudaPitchedPtr bsplineCoeffs = CastVolumeHostToDevice(voxels, volumeSize.x, volumeSize.y, volumeSize.z);
	CubicBSplinePrefilter3DTimer((float*)bsplineCoeffs.ptr, (uint)bsplineCoeffs.pitch, volumeSize.x, volumeSize.y, volumeSize.z);

	// create the b-spline coefficients texture
	cudaArray *coeffArray = 0;
	cudaExtent volumeExtent = make_cudaExtent(volumeSize.x, volumeSize.y, volumeSize.z);
	CreateTextureFromVolume(&coeffs, &coeffArray, bsplineCoeffs, volumeExtent, true);
	CUDA_SAFE_CALL(cudaFree(bsplineCoeffs.ptr));  //they are now in the coeffs texture, we do not need this anymore

	// Now create a texture with the original sample values for nearest neighbor and linear interpolation
	// Note that if you are going to do cubic interpolation only, you can remove the following code
	cudaArray *volumeArray = 0;
	CreateTextureFromVolume(&tex, &volumeArray, voxels, volumeExtent, false);
}
