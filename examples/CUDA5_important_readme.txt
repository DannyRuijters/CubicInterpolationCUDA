To build the examples with CUDA5 on windows, perform the following steps:

1) In Visual Studio 2008 or 2010 create a new CUDA5 project: File->New->Project...->NVIDIA->CUDA->CUDA 5.0 Runtime
2) Give the project some logical name (e.g. "cudaCubicTexture3D_cuda5")
3) Remove the default template "kernel.cu" from the project
4) Move the .sln and .vcproj files to the example directory you want to compile
5) Add the .cpp and .cu files of the example: Project->Add Existing Item...
6) Project->Properties->Configuration Properties->C/C++->Code Generation->Runtime Library: Set to "Multi-threaded (/MT)"
7) Project->Properties->Configuration Properties->C/C++->General->Additional Include Directories: Add "../cuda5_fix;$(CUDA_PATH_V5_0)\include"
8) Project->Properties->Configuration Properties->CUDA Runtime API->General->Additional Include Directories: Add "../cuda5_fix;../../code"
9) Project->Properties->Configuration Properties->Linker>Input->Additional Dependencies: Add "../cuda5_fix/lib/glut32.lib ../cuda5_fix/lib/glew32.lib"
10) Project->Properties->Configuration Properties->General->Output Directory: Set to "../../bin/win32/release"
11) Project->Properties->Configuration Properties->Debugging->Command Arguments: Set to something appropriate, e.g. "../data/Bucky.raw 32 32 32"


To build the examples with CUDA5 on Linux/Mac, use the Makefile_cuda5:
make -f Makefile_cuda5
