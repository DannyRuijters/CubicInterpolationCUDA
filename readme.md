CUDA Cubic B-Spline Interpolation (CI)
======================================

Version 1.2
-----------

This *read me* serves as a quick guide to using the CUDA Cubic B-Spline
Interpolation (abbreviated as CI) code. The most recent version of CI
and some background information can be found
[online](http://www.dannyruijters.nl/cubicinterpolation/). To execute
and compile CI you need [CUDA and the CUDA SDK (2.0 or
higher)](http://www.nvidia.com/object/cuda_get.html). Read
[this](examples/CUDA5_important_readme.txt), if you are using CUDA 5.
This software has been released under a revised BSD style
[license](license.txt).

Getting started
---------------

If you want to simply replace linear interpolation by cubic filtering,
then all you need to do is to include the appropriate header and replace
your `tex1D`, `tex2D` and `tex3D` calls by one of the following:

1D textures:

 `cubicTex1D(texture tex, float x)`

2D textures:

 `cubicTex2D(texture tex, float x, float y)`  
 `cubicTex2D(texture tex, float2 coord)`

3D textures:

 `cubicTex3D(texture tex, float x, float y, float z)`  
 `cubicTex3D(texture tex, float3 coord)`

Whereby the texture coordinates `coord` are expressed in absolute pixel
respectively voxel coordinates (thus not in normalized coordinates).

It is also possible to query the cubicly filtered 1st order derivative
of the texture at a coordinate `coord`; e.g. in the x-direction: 

 `cubicTex3D_1st_derivative_x(texture tex, float3 coord)`

The calls for the y- and z-direction are similar, and also the
derivatives of 1D and 2D textures can be retrieved in the same way. The
cost of querying the derivative in a single direction costs the same
amount of time as a normal cubicly filtered lookup in the texture. The
gradient of the texture at a specified coordinate can be composed by
querying the derivatives in x-, y- and z-direction at that location.

Pre-filtering
-------------

When the approach described above is directly applied, it will result in
smoothened images. This is caused by the fact that the cubic B-spline
filtering yields a function that does not pass through its coefficients
(i.e. texture values). In order to wind up with a cubic B-spline
interpolated image that passes through the original samples, we need to
pre-filter the texture, as is beautifully described by [Philippe
Thévenaz *et al.*](http://bigwww.epfl.ch/thevenaz/interpolation/)

Luckily, CI also provides a CUDA implementation of this prefilter, and
using it is rather simple. The interface for the 2D case is:

`CubicBSplinePrefilter2D(float* image, uint pitch, uint width, uint height);`

and for the 3D case:

`CubicBSplinePrefilter3D(float* volume, uint pitch, uint width, uint height, uint depth);`

The `image` and `volume` should point to the GPU memory containing the
data samples. Note that the sample values will be replaced by the cubic
B-spline coefficients. The `pitch` variable describes the width of a row
in the image in bytes. `width`, `height`, and `depth` describe the
extent of the data in pixels/voxels. Instead of `float*`, it is also
possible to pass `float2*`, `float3*`, or `float4*` data (e.g., for RGB
color data).

Getting your data there
-----------------------

In order to make your even easier, CI also provides some routines to
transfer your data to the GPU memory, and back. Copying your sample
values to the GPU memory can be accomplished by this function:

`cudaPitchedPtr CopyVolumeHostToDevice(const float* host, uint width, uint height, uint depth);`

The routine allocates GPU memory and copies the sample values from CPU
to GPU memory. The pointer to the CPU memory is passed as `host`, and a
pointer to the GPU memory is returned. The counterpart that copies data
from the GPU memory to the CPU memory is called:

`void CopyVolumeDeviceToHost(float* host, cudaPitchedPtr device, uint width, uint height, uint depth);`

Note that the `host` destination CPU memory should be allocated before
`CopyVolumeDeviceToHost` is called. This routine will also free the GPU
memory.

Often you will have your original data in a different format than
`float`, and for large data sets it costs some time (and memory) to cast
everything to `float`. Therefore CI also provides a number of functions
that copy and cast your data immediately to `float` on the GPU, which is
faster and easier. In C++ you can use the following template function:

`template extern cudaPitchedPtr CastVolumeHostToDevice(const T* host, uint width, uint height, uint depth);`

The usage of the parameters is the same as for `CopyVolumeHostToDevice`,
except that `host` can be of the type `uchar`, `schar`, `ushort` or
`short`. Note that the sample values will be normalized, meaning that
the maximum value (e.g. 255 for `uchar`) will be mapped to 1.0.

The following function can be used to copy the output of the pre-filter
functions into a texture:

`void CreateTextureFromVolume(texture* tex, cudaArray** texArray, const cudaPitchedPtr volume, cudaExtent extent, bool onDevice);`

Using layered textures
----------------------

It is possible to use the cubic interpolation texture lookups with
layered textures. In order to do this, you need to add:  
 `#define cudaTextureType2DLayered`  
 to your code, and replace the calls to:  
 `tex2DLayered(texture tex, float x, float y, int layer)`  
 with:  
 `cubicTex2DLayered(texture tex, float x, float y, int layer)`  
 For 1D it works similarly.

Example programs
----------------

Some example programs are provided along with the CI code, to illustrate
the usage of the various routines. In order to compile the example
programs, you first need to make sure that the [CUDA SDK
examples](http://www.nvidia.com/object/cuda_get.html) can be compiled.
Then it is simply a matter of opening the Visual Studio solution on
Windows machines, or running `make` in the example folder on the Mac or
Linux.

-   **cudaCubicRotate2D** uses 2D cubic interpolation to apply a user
    defined rotation and translation to a given image. The code is based
    on the example program provided by [Philippe
    Thévenaz](http://bigwww.epfl.ch/thevenaz/interpolation/), and is
    adapted to make use of the CI routines. The main program is in C
    rather than in C++, and therefore shows how to exploit CUDA in
    general and CI in particular from within C code. When comparing the
    resulting images with the counterparts generated with the original
    source code, it becomes immediately apparent that presently CI does
    not support mirroring periodicity when exceeding the image extent.
-   **cudaCubicTexture3D** is an adapted version of the simpleTexture3D
    that can be found in the CUDA SDK 2.0. It shows how the CI code can
    be used to pre-filter the texture data, and how to perform the cubic
    interpolation. This example program also illustrates the cubic
    interpolation image quality compared to nearest neighbor, linear and
    non pre-filtered interpolation, since it can switch between those on
    the fly by pressing the 'f' key.
-   **referenceCubicTexture3D** performs the same task as
    simpleCubicTexture3D, but uses the CPU instead of the GPU. It is
    meant to generate a ground truth for performance measurements, and
    allows you to repeat those measurements on your own hardware.
-   **cudaCubicRayCast** is a very simple CUDA raycasting program that
    demonstrates the merits of cubic interpolation (including
    prefiltering) in 3D volume rendering.
-   **glCubicRayCast** shows raycasting with cubic interpolation using
    pure OpenGL, without CUDA. However, this example also lacks the
    prefiltering of the voxel data.
-   **cudaCubicAviPlayback** prefilters every frame of an AVI movie on
    the fly. This example illustrates how the CI code can be used with
    vector data (such as RGBA color data).
-   **cudaAccuracyTest** creates 1D, 2D and 3D prefiltered textures, and
    uses the cubic interpolation routines to reconstruct the values at
    the sample knots. The cubicly interpolated values are then compared
    to the original (randomly generated) sample values.

Background
----------

More background information to the CI code is provided
[online](http://www.dannyruijters.nl/cubicinterpolation/). A
comprehensive discussion of uniform B-spline interpolation and the
pre-filter can be found in [1]. The GPU implementation is described in
[2]. The fast cubic B-spline interpolation is an adapted version of the
method introduced by Sigg and Hadwiger [3]. A description of the adapted
algorithm, its merits and its drawbacks is given in [4].

1.  Philippe Thévenaz, Thierry Blu, and Michael Unser, "[Interpolation
    Revisited](http://bigwww.epfl.ch/publications/thevenaz0002.pdf),"
    IEEE Transactions on Medical Imaging, vol. 19, no. 7, pp. 739-758,
    July 2000.
2.  Daniel Ruijters and Philippe Thévenaz, "[GPU Prefilter for Accurate
    Cubic B-Spline
    Interpolation](http://dannyruijters.nl/docs/cudaPrefilter3.pdf),"
    The Computer Journal, vol. 55, no. 1, pp. 15-20, January 2012.
3.  Christian Sigg and Markus Hadwiger, "[Fast Third-Order Texture
    Filtering](http://developer.download.nvidia.com/SDK/9.5/Samples/DEMOS/OpenGL/src/fast_third_order/docs/Gems2_ch20_SDK.pdf),"
    In GPU Gems 2: Programming Techniques for High-Performance Graphics
    and General-Purpose Computation, Matt Pharr (ed.), Addison-Wesley;
    chapter 20, pp. 313-329, 2005.
4.  Daniel Ruijters, Bart M. ter Haar Romeny, and Paul Suetens,
    "[Efficient GPU-Based Texture Interpolation using Uniform
    B-Splines](http://www.mate.tue.nl/mate/pdfs/10318.pdf)," Journal of
    Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.

Copyright 2008-2013 Danny Ruijters
