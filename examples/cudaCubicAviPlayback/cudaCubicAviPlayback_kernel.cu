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
#include <castFloat4.cu>
#include <cubicPrefilter2D.cu>
#include <cubicTex2D.cu>

cudaArray *coeffArray = 0;
texture<float4, 2, cudaReadModeElementType> coeffs;  //2D texture


__global__ void
render_kernel(uchar4* output, uint width, float2 image2frame, uint filterMethod)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	float2 coord = image2frame * make_float2(x, y);

	// read from 2D texture
	float4 texel;
	switch (filterMethod)
	{
		case 0:  //nearest neighbor
		case 1: texel = tex2D(coeffs, coord.x, coord.y); break;  //linear
		case 2: texel = cubicTex2D(coeffs, coord.x, coord.y); break;  //fast cubic
		case 3: texel = cubicTex2D(coeffs, coord.x, coord.y); break;  //non-prefiltered, fast cubic
	}

	// write output color
	texel = 255.0f * make_float4(__saturatef(texel.x), __saturatef(texel.y), __saturatef(texel.z), __saturatef(texel.w));
	uint i = __umul24(y, width) + x;
	output[i] = make_uchar4(texel.z, texel.y, texel.x, texel.w); //BGRA
}


// render image using CUDA
extern "C" void render(uchar4* output, uint2 windowExtent, uint2 frameExtent, uint filterMethod)
{
	// set texture parameters
	coeffs.filterMode = (filterMethod == 0) ? cudaFilterModePoint : cudaFilterModeLinear;

	// call CUDA kernel, writing results to PBO
	const dim3 blockSize(min(PowTwoDivider(windowExtent.x), 16), min(PowTwoDivider(windowExtent.y), 16));
	const dim3 gridSize(windowExtent.x / blockSize.x, windowExtent.y / blockSize.y);
	const float2 image2frame = make_float2((float)frameExtent.x, (float)frameExtent.y) / make_float2((float)windowExtent.x, (float)windowExtent.y);
	render_kernel<<<gridSize, blockSize>>>(output, windowExtent.x, image2frame, filterMethod);
	CUT_CHECK_ERROR("kernel failed");
}


extern "C" void initTexture(const uchar* rgb, uint width, uint height, uint filterMethod)
{
	cudaPitchedPtr image = CastVolumeHost3ToDevice4(rgb, width, height, 1);
	if (filterMethod == 2) CubicBSplinePrefilter2DTimer<float4>((float4*)image.ptr, (uint)image.pitch, width, height);

	// Create the B-spline coefficients texture
	cudaChannelFormatDesc channelDescCoeff = cudaCreateChannelDesc<float4>();
	if (coeffArray == 0) CUDA_SAFE_CALL(cudaMallocArray(&coeffArray, &channelDescCoeff, width, height));
	CUDA_SAFE_CALL(cudaMemcpy2DToArray(coeffArray, 0, 0, image.ptr, image.pitch, width * sizeof(float4), height, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaFree(image.ptr));
	CUDA_SAFE_CALL(cudaBindTextureToArray(coeffs, coeffArray, channelDescCoeff));
	coeffs.normalized = false;  // access with normalized texture coordinates
	coeffs.filterMode = cudaFilterModeLinear;
}


