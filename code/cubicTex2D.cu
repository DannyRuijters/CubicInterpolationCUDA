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

#ifndef _CUBIC2D_KERNEL_H_
#define _CUBIC2D_KERNEL_H_

#include "internal/bspline_kernel.cu"

//! Bilinearly interpolated texture lookup, using unnormalized coordinates.
//! This function merely serves as a reference for the bicubic versions.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
template<class T, enum cudaTextureReadMode mode>
__device__ float linearTex2D(texture<T, 2, mode> tex, float x, float y)
{
	return tex2D(tex, x, y);
}

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Straight forward implementation, using 16 nearest neighbour lookups.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
template<class T, enum cudaTextureReadMode mode>
__device__ float cubicTex2DSimple(texture<T, 2, mode> tex, float x, float y)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float2 coord_grid = make_float2(x - 0.5f, y - 0.5f);
	float2 index = floor(coord_grid);
	const float2 fraction = coord_grid - index;
	index.x += 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]
	index.y += 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

	float result = 0.0f;
	for (float y=-1; y < 2.5f; y++)
	{
		float bsplineY = bspline(y-fraction.y);
		float v = index.y + y;
		for (float x=-1; x < 2.5f; x++)
		{
			float bsplineXY = bspline(x-fraction.x) * bsplineY;
			float u = index.x + x;
			result += bsplineXY * tex2D(tex, u, v);
		}
	}
	return result;
}

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 4 trilinear lookups.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
#define WEIGHTS bspline_weights
#define CUBICTEX2D cubicTex2D
#include "internal/cubicTex2D_kernel.cu"
#undef CUBICTEX2D
#undef WEIGHTS

// Fast bicubic interpolated 1st order derivative texture lookup in x- and
// y-direction, using unnormalized coordinates.
__device__ void bspline_weights_1st_derivative_x(float2 fraction, float2& w0, float2& w1, float2& w2, float2& w3)
{
	float t0, t1, t2, t3;
	bspline_weights_1st_derivative(fraction.x, t0, t1, t2, t3);
	w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
	bspline_weights(fraction.y, t0, t1, t2, t3);
	w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
}

__device__ void bspline_weights_1st_derivative_y(float2 fraction, float2& w0, float2& w1, float2& w2, float2& w3)
{
	float t0, t1, t2, t3;
	bspline_weights(fraction.x, t0, t1, t2, t3);
	w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
	bspline_weights_1st_derivative(fraction.y, t0, t1, t2, t3);
	w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
}

#define WEIGHTS bspline_weights_1st_derivative_x
#define CUBICTEX2D cubicTex2D_1st_derivative_x
#include "internal/cubicTex2D_kernel.cu"
#undef CUBICTEX2D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative_y
#define CUBICTEX2D cubicTex2D_1st_derivative_y
#include "internal/cubicTex2D_kernel.cu"
#undef CUBICTEX2D
#undef WEIGHTS


#ifdef cudaTextureType2DLayered
// support for layered texture calls
#undef _EXTRA_ARGS
#undef _PASS_EXTRA_ARGS
#undef _TEX2D
#undef _TEXTYPE
#define _EXTRA_ARGS , int layer
#define _PASS_EXTRA_ARGS , layer
#define _TEX2D tex2DLayered
#define _TEXTYPE cudaTextureType2DLayered

#define WEIGHTS bspline_weights
#define CUBICTEX2D cubicTex2DLayered
#include "internal/cubicTex2D_kernel.cu"
#undef CUBICTEX2D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative_x
#define CUBICTEX2D cubicTex2DLayered_1st_derivative_x
#include "internal/cubicTex2D_kernel.cu"
#undef CUBICTEX2D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative_y
#define CUBICTEX2D cubicTex2DLayered_1st_derivative_y
#include "internal/cubicTex2D_kernel.cu"
#undef CUBICTEX2D
#undef WEIGHTS
#endif //cudaTextureType2DLayered

#undef _EXTRA_ARGS
#undef _PASS_EXTRA_ARGS
#undef _TEX2D
#undef _TEXTYPE

#endif // _CUBIC2D_KERNEL_H_
