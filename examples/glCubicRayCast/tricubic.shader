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


const char tricubic[] =
//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 8 trilinear lookups.
//! @param tex  3D texture
//! @param coord  normalized 3D texture coordinate
//"uniform vec3 nrOfVoxels;\n"
"float interpolate_tricubic_fast(sampler3D tex, vec3 coord)\n"
"{\n"
	// shift the coordinate from [0,1] to [-0.5, nrOfVoxels-0.5]
"	vec3 nrOfVoxels = vec3(textureSize3D(tex, 0));\n"
"	vec3 coord_grid = coord * nrOfVoxels - 0.5;\n"
"	vec3 index = floor(coord_grid);\n"
"	vec3 fraction = coord_grid - index;\n"
"	vec3 one_frac = 1.0 - fraction;\n"

"	vec3 w0 = 1.0/6.0 * one_frac*one_frac*one_frac;\n"
"	vec3 w1 = 2.0/3.0 - 0.5 * fraction*fraction*(2.0-fraction);\n"
"	vec3 w2 = 2.0/3.0 - 0.5 * one_frac*one_frac*(2.0-one_frac);\n"
"	vec3 w3 = 1.0/6.0 * fraction*fraction*fraction;\n"

"	vec3 g0 = w0 + w1;\n"
"	vec3 g1 = w2 + w3;\n"
"	vec3 mult = 1.0 / nrOfVoxels;\n"
"	vec3 h0 = mult * ((w1 / g0) - 0.5 + index);  //h0 = w1/g0 - 1, move from [-0.5, nrOfVoxels-0.5] to [0,1]\n"
"	vec3 h1 = mult * ((w3 / g1) + 1.5 + index);  //h1 = w3/g1 + 1, move from [-0.5, nrOfVoxels-0.5] to [0,1]\n"

	// fetch the eight linear interpolations
	// weighting and fetching is interleaved for performance and stability reasons
"	float tex000 = texture3D(tex, h0).r;\n"
"	float tex100 = texture3D(tex, vec3(h1.x, h0.y, h0.z)).r;\n"
"	tex000 = mix(tex100, tex000, g0.x);  //weigh along the x-direction\n"
"	float tex010 = texture3D(tex, vec3(h0.x, h1.y, h0.z)).r;\n"
"	float tex110 = texture3D(tex, vec3(h1.x, h1.y, h0.z)).r;\n"
"	tex010 = mix(tex110, tex010, g0.x);  //weigh along the x-direction\n"
"	tex000 = mix(tex010, tex000, g0.y);  //weigh along the y-direction\n"
"	float tex001 = texture3D(tex, vec3(h0.x, h0.y, h1.z)).r;\n"
"	float tex101 = texture3D(tex, vec3(h1.x, h0.y, h1.z)).r;\n"
"	tex001 = mix(tex101, tex001, g0.x);  //weigh along the x-direction\n"
"	float tex011 = texture3D(tex, vec3(h0.x, h1.y, h1.z)).r;\n"
"	float tex111 = texture3D(tex, h1).r;\n"
"	tex011 = mix(tex111, tex011, g0.x);  //weigh along the x-direction\n"
"	tex001 = mix(tex011, tex001, g0.y);  //weigh along the y-direction\n"

"	return mix(tex001, tex000, g0.z);  //weigh along the z-direction\n"
"}\n";
