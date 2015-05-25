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

#ifndef _CAST_FLOAT4_H_
#define _CAST_FLOAT4_H_

#include "memcpy.cu"


//--------------------------------------------------------------------------
// Declare the interleaved copu CUDA kernel
//--------------------------------------------------------------------------
template<class T> __global__ void CopyCastInterleaved(uchar* destination, const T* source, uint pitch, uint width)
{
	uint2 index = make_uint2(
		__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
		__umul24(blockIdx.y, blockDim.y) + threadIdx.y);
	uint index3 = 3 * (index.y * width + index.x);
	
	float4* dest = (float4*)(destination + index.y * pitch) + index.x;
	float mult = 1.0f / Multiplier<T>();
	*dest = make_float4(
		mult * (float)source[index3],
		mult * (float)source[index3+1],
		mult * (float)source[index3+2], 1.0f);
}

//--------------------------------------------------------------------------
// Declare the typecast templated function
// This function can be called directly in C++ programs
//--------------------------------------------------------------------------

//! Allocate GPU memory and copy a voxel volume from CPU to GPU memory
//! and cast it to the normalized floating point format
//! @return the pointer to the GPU copy of the voxel volume
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
template<class T> extern cudaPitchedPtr CastVolumeHost3ToDevice4(const T* host, uint width, uint height, uint depth)
{
	cudaPitchedPtr device = {0};
	const cudaExtent extent = make_cudaExtent(width * sizeof(float4), height, depth);
	CUDA_SAFE_CALL(cudaMalloc3D(&device, extent));
	const size_t pitchedBytesPerSlice = device.pitch * device.ysize;
	
	T* temp = 0;
	const uint voxelsPerSlice = width * height;
	const size_t nrOfBytesTemp = voxelsPerSlice * 3 * sizeof(T);
	CUDA_SAFE_CALL(cudaMalloc((void**)&temp, nrOfBytesTemp));

	uint dimX = min(PowTwoDivider(width), 64);
	dim3 dimBlock(dimX, min(PowTwoDivider(height), 512 / dimX));
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
	size_t offsetHost = 0;
	size_t offsetDevice = 0;
	
	for (uint slice = 0; slice < depth; slice++)
	{
		CUDA_SAFE_CALL(cudaMemcpy(temp, host + offsetHost, nrOfBytesTemp, cudaMemcpyHostToDevice));
		CopyCastInterleaved<T><<<dimGrid, dimBlock>>>((uchar*)device.ptr + offsetDevice, temp, (uint)device.pitch, width);
		CUT_CHECK_ERROR("Cast kernel failed");
		offsetHost += voxelsPerSlice;
		offsetDevice += pitchedBytesPerSlice;
	}

	CUDA_SAFE_CALL(cudaFree(temp));  //free the temp GPU volume
	return device;
}

#endif  //_CAST_FLOAT4_H_
