/*--------------------------------------------------------------------------*\
Copyright (c) 2012-2012, Danny Ruijters. All rights reserved.
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

// Accuracy test
//
// This sample creates 1D, 2D and 3D prefiltered textures, and uses the
// cubic interpolation routines to reconstruct the values at the sample
// knots. The cubicly interpolated values are then compared to the original
// (randomly generated) sample values.


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cutil.h>

extern "C" void test(uint3 volumeSize);
extern "C" void initCuda(uint3 volumeSize);
extern "C" void freeCuda();


uint3 GetVolumeSize()
{
	uint3 volumeSize;
	printf("\nEnter number of samples in x-direction: ");
    scanf("%d", &volumeSize.x);
	printf("Enter number of samples in y-direction: ");
    scanf("%d", &volumeSize.y);
	printf("Enter number of samples in z-direction: ");
    scanf("%d", &volumeSize.z);
	return volumeSize;
}

int main(int argc, char** argv) 
{
	CUT_DEVICE_INIT(argc, argv);

	uint3 volumeSize = GetVolumeSize();
	while (volumeSize.x > 0 && volumeSize.y > 0 && volumeSize.z > 0)
	{
		initCuda(volumeSize);
		test(volumeSize);
		freeCuda();
		volumeSize = GetVolumeSize();
	}

    return 0;
}
