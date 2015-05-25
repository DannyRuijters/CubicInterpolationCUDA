/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2013, Danny Ruijters. All rights reserved.
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

#ifndef _CUBIC1D_KERNEL_H_
#define _CUBIC1D_KERNEL_H_

#include "internal/bspline_kernel.cu"

//! Linearly interpolated texture lookup, using unnormalized coordinates.
//! This function merely serves as a reference for the cubic versions.
//! @param tex  1D texture
//! @param x  unnormalized x texture coordinate
template<class T, enum cudaTextureReadMode mode>
__device__ float linearTex1D(texture<T, 1, mode> tex, float x)
{
	return tex1D(tex, x);
}

//! Cubic interpolated texture lookup, using unnormalized coordinates.
//! Straight forward implementation, using 4 nearest neighbour lookups.
//! @param tex  1D texture
//! @param x  unnormalized x texture coordinate
template<class T, enum cudaTextureReadMode mode>
__device__ float cubicTex1DSimple(texture<T, 1, mode> tex, float x)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float coord_grid = x - 0.5f;
	float index = floor(coord_grid);
	const float fraction = coord_grid - index;
	index += 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

	float result = 0.0f;
	for (float x=-1; x < 2.5f; x++)
	{
		float bsplineX = bspline(x-fraction);
		float u = index + x;
		result += bsplineX * tex1D(tex, u);
	}
	return result;
}

//! Cubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 2 linear lookups.
//! @param tex  1D texture
//! @param x  unnormalized x texture coordinate
#define WEIGHTS bspline_weights
#define CUBICTEX1D cubicTex1D
#include "internal/cubicTex1D_kernel.cu"
#undef CUBICTEX1D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative
#define CUBICTEX1D cubicTex1D_1st_derivative
#include "internal/cubicTex1D_kernel.cu"
#undef CUBICTEX1D
#undef WEIGHTS


#ifdef cudaTextureType1DLayered
// support for layered texture calls
#undef _EXTRA_ARGS
#undef _PASS_EXTRA_ARGS
#undef _TEX1D
#undef _TEXTYPE
#define _EXTRA_ARGS , int layer
#define _PASS_EXTRA_ARGS , layer
#define _TEX1D tex1DLayered
#define _TEXTYPE cudaTextureType1DLayered

#define WEIGHTS bspline_weights
#define CUBICTEX1D cubicTex1DLayered
#include "internal/cubicTex1D_kernel.cu"
#undef CUBICTEX1D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative
#define CUBICTEX1D cubicTex1DLayered_1st_derivative
#include "internal/cubicTex1D_kernel.cu"
#undef CUBICTEX1D
#undef WEIGHTS
#endif // cudaTextureType1DLayered

#undef _EXTRA_ARGS
#undef _PASS_EXTRA_ARGS
#undef _TEX1D
#undef _TEXTYPE

#endif // _CUBIC1D_KERNEL_H_
