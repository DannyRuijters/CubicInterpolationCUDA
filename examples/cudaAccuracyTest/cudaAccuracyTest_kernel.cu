/*--------------------------------------------------------------------------*\
Copyright (c) 2012-2013, Danny Ruijters. All rights reserved.
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
#include <cubicPrefilter3D.cu>
#include <cubicTex1D.cu>
#include <cubicTex2D.cu>
#include <cubicTex3D.cu>

texture<float, 1, cudaReadModeElementType> coeffs1D;  //1D texture
texture<float, 2, cudaReadModeElementType> coeffs2D;  //2D texture
texture<float, 3, cudaReadModeElementType> coeffs3D;  //3D texture
float* input = NULL;
float* outputCuda = NULL;
cudaArray* coeffArray1D = 0;
cudaArray* coeffArray2D = 0;
cudaArray* coeffArray3D = 0;


__global__ void kernel1D(float* output, float rSize)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	output[x] = cubicTex1D(coeffs1D, (float)x+0.5f);
}

__global__ void kernel2D(float* output, int2 extent)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	float2 coord = make_float2(x+0.5f, y+0.5f);

	uint i = y * extent.x + x;
	output[i] = cubicTex2D(coeffs2D, coord);
}

__global__ void kernel3D(float* output, int3 extent, uint z)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	float3 coord = make_float3(x+0.5f, y+0.5f, z+0.5f);

	uint i = y * extent.x + x;  //index in the output slice
	output[i] = cubicTex3D(coeffs3D, coord);
}

static double Diff(float* a, float* b, int size)
{
	double sum = 0.0;
	for (int i=0; i < size; i++, a++, b++)
	{
		double diff = *a - *b;
		sum += fabs(diff);
	}
	
	return sum / size;
}

// render image using CUDA
extern "C" void test(uint3 volumeSize)
{
	const int size = volumeSize.x * volumeSize.y;
	float* output = new float[size];

	// call CUDA kernel
	const dim3 blockSize1(min(PowTwoDivider(volumeSize.x), 64), 1);
	const dim3 gridSize1(volumeSize.x / blockSize1.x, 1);
	kernel1D<<<gridSize1, blockSize1>>>(outputCuda, 1.0f/volumeSize.x);
	CUT_CHECK_ERROR("kernel failed");
	CUDA_SAFE_CALL(cudaMemcpy(output, outputCuda, volumeSize.x * sizeof(float), cudaMemcpyDeviceToHost));
	printf("1D mean absolute error: %f\n", Diff(input, output, volumeSize.x));
	
	const dim3 blockSize2(min(PowTwoDivider(volumeSize.x), 16), min(PowTwoDivider(volumeSize.y), 16));
	const dim3 gridSize2(volumeSize.x / blockSize2.x, volumeSize.y / blockSize2.y);
	kernel2D<<<gridSize2, blockSize2>>>(outputCuda, make_int2(volumeSize.x, volumeSize.y));
	CUT_CHECK_ERROR("kernel failed");
	CUDA_SAFE_CALL(cudaMemcpy(output, outputCuda, size * sizeof(float), cudaMemcpyDeviceToHost));
	printf("2D mean absolute error: %f\n", Diff(input, output, size));
	
	double sum = 0.0;
	const int3 volumeExtent = make_int3(volumeSize.x, volumeSize.y, volumeSize.z);
	for (uint z=0; z < volumeSize.z; z++)
	{
		kernel3D<<<gridSize2, blockSize2>>>(outputCuda, volumeExtent, z);
		CUT_CHECK_ERROR("kernel failed");
		CUDA_SAFE_CALL(cudaMemcpy(output, outputCuda, size * sizeof(float), cudaMemcpyDeviceToHost));
		sum += Diff(input + z * size, output, size);
	}
	
	delete[] output;
	printf("3D mean absolute error: %f\n", (sum/volumeSize.z));
}


// intialize the textures, and calculate the cubic B-spline coefficients
extern "C" void initCuda(uint3 volumeSize)
{
	// initialize volume with random values between 0.0 and 1.0
	srand((unsigned)time(0));
	const int size = volumeSize.x * volumeSize.y * volumeSize.z;
	input = new float[size];
	for (int i=0; i < size; i++)
	{
		input[i] = (float)rand()/(float)RAND_MAX;
	}

	// initialize CUDA output array
	CUDA_SAFE_CALL(cudaMalloc((void**)&outputCuda, volumeSize.x * volumeSize.y * sizeof(float)));
	
	// 1D Texture
	float* line = new float[volumeSize.x];
	memcpy(line, input, volumeSize.x * sizeof(float));
	ConvertToInterpolationCoefficients(line, volumeSize.x, sizeof(float));
	cudaPitchedPtr bsplineCoeffs1D = CopyVolumeHostToDevice(line, volumeSize.x, 1, 1);
	delete[] line;
	
	cudaChannelFormatDesc channelDescCoeff = cudaCreateChannelDesc<float>();
	CUDA_SAFE_CALL(cudaMallocArray(&coeffArray1D, &channelDescCoeff, volumeSize.x, 1));
	CUDA_SAFE_CALL(cudaMemcpyToArray(coeffArray1D, 0, 0, bsplineCoeffs1D.ptr, volumeSize.x * sizeof(float), cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaFree(bsplineCoeffs1D.ptr));  //they are now in the coeffs texture, we do not need this anymore
	CUDA_SAFE_CALL(cudaBindTextureToArray(coeffs1D, coeffArray1D, channelDescCoeff));
	coeffs1D.normalized = false;  //access with unnormalized texture coordinates
	coeffs1D.filterMode = cudaFilterModeLinear;
	
	// 2D Texture	
	cudaPitchedPtr bsplineCoeffs2D = CopyVolumeHostToDevice(input, volumeSize.x, volumeSize.y, 1);
	CubicBSplinePrefilter2DTimer((float*)bsplineCoeffs2D.ptr, (uint)bsplineCoeffs2D.pitch, volumeSize.x, volumeSize.y);
	CUDA_SAFE_CALL(cudaMallocArray(&coeffArray2D, &channelDescCoeff, volumeSize.x, volumeSize.y));
	CUDA_SAFE_CALL(cudaMemcpy2DToArray(coeffArray2D, 0, 0, bsplineCoeffs2D.ptr, bsplineCoeffs2D.pitch, volumeSize.x * sizeof(float), volumeSize.y, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaFree(bsplineCoeffs2D.ptr));  //they are now in the coeffs texture, we do not need this anymore
	CUDA_SAFE_CALL(cudaBindTextureToArray(coeffs2D, coeffArray2D, channelDescCoeff));
	//CUDA_SAFE_CALL(cudaBindTexture2D(NULL, coeffs2D, bsplineCoeffs2D.ptr, volumeSize.x, volumeSize.y, bsplineCoeffs2D.pitch));  //on recent CUDA versions, this call can replace the four previous ones
	coeffs2D.normalized = false;  //access with unnormalized texture coordinates
	coeffs2D.filterMode = cudaFilterModeLinear;
	
	// 3D Texture
	// calculate the b-spline coefficients
	cudaPitchedPtr bsplineCoeffs3D = CopyVolumeHostToDevice(input, volumeSize.x, volumeSize.y, volumeSize.z);
	CubicBSplinePrefilter3DTimer((float*)bsplineCoeffs3D.ptr, (uint)bsplineCoeffs3D.pitch, volumeSize.x, volumeSize.y, volumeSize.z);
	// create the b-spline coefficients texture
	cudaExtent volumeExtent = make_cudaExtent(volumeSize.x, volumeSize.y, volumeSize.z);
	CreateTextureFromVolume(&coeffs3D, &coeffArray3D, bsplineCoeffs3D, volumeExtent, true);
	CUDA_SAFE_CALL(cudaFree(bsplineCoeffs3D.ptr));  //they are now in the coeffs texture, we do not need this anymore
}


extern "C" void freeCuda()
{
	CUDA_SAFE_CALL(cudaFree(outputCuda));
	CUDA_SAFE_CALL(cudaFreeArray(coeffArray1D));
	CUDA_SAFE_CALL(cudaFreeArray(coeffArray2D));
	CUDA_SAFE_CALL(cudaFreeArray(coeffArray3D));
	delete[] input;
	
	outputCuda = NULL;
	coeffArray3D = NULL;
	input = NULL;
}
