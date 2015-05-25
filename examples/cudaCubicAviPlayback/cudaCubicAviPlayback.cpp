/*--------------------------------------------------------------------------*\
Copyright (c) 2010, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/

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


// This sample loads an AVI movie from disk and displays it
// using 2D texture lookups. The interpolation can be switched between
// nearest neighbor, linear, and pre-filtered fast cubic b-spline.

#include <string>
#include <GL/glew.h>
#if defined(_WIN32)
#include <windows.h>
#include <vfw.h>
#include <GL/wglew.h>
#endif

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_runtime.h>
#include <cutil.h>
#include <cuda_gl_interop.h>

typedef unsigned int uint;
typedef unsigned char uchar;

uint2 frameExtent;
uint2 windowExtent = make_uint2(512, 512);  //output window
GLuint pbo = 0;  //OpenGL pixel buffer object
uint hTimer;
uint filterMethod = 2;
bool animate = true;

// AVI variables
PAVIFILE pAviFile = NULL;
LPBITMAPINFO pChunk = NULL;
PGETFRAME pgf = NULL;
LONG endFrame = 0;
float framerate = 24.0f;

extern "C" void render(uchar4* output, uint2 windowExtent, uint2 frameExtent, uint filterMethod);
extern "C" void initTexture(const uchar* rgb, uint width, uint height, uint filterMethod);

void LoadAVI(LPCTSTR lpszPathName)
{
	// Step 1: initialize AVI engine
	AVIFileInit();


	// Step 2: open the movie file for reading....
	HRESULT hr = AVIFileOpen(&pAviFile, lpszPathName, OF_READ, NULL);

	if (hr != AVIERR_OK)
	{
		switch(hr)
		{
		case AVIERR_BADFORMAT:
			throw("AVI Engine failed to initialize\nThe file couldn't be read, indicating a corrupt file or an unrecognized format.\n");
			break;
		case AVIERR_MEMORY:
			throw("AVI Engine failed to initialize\nThe file could not be opened because of insufficient memory.\n");
			break;
		case AVIERR_FILEREAD:
			throw("AVI Engine failed to initialize\nA disk error occurred while reading the file.\n");
			break;
		case AVIERR_FILEOPEN:
			throw("AVI Engine failed to initialize\nA disk error occurred while opening the file.\n");
			break;
		case REGDB_E_CLASSNOTREG:
			throw("AVI Engine failed to initialize\nAccording to the registry, the type of file specified in AVIFileOpen does not have a handler to process it.\n");
			break;
		default:
			throw("AVI Engine failed to initialize\nUnhandeld error.\n");
			break;
		}
	}


	// Step 3: open AVI streams
	const int MAX_VIDEO_STREAMS = 5;
	PAVISTREAM pVideo[MAX_VIDEO_STREAMS];
	int nNumVideoStreams = 0;

	do
	{
	   if (AVIFileGetStream(pAviFile, &pVideo[nNumVideoStreams], streamtypeVIDEO, nNumVideoStreams)) break;
	   if (pVideo[nNumVideoStreams] == NULL) break;
	}
	while (++nNumVideoStreams < MAX_VIDEO_STREAMS);

	
	// Step 4: loop over first stream
	for (int nStream = 0; nStream < 1 /*nNumVideoStreams*/; ++nStream)
	{
		AVISTREAMINFO info;
		if (AVIStreamInfo(pVideo[nStream], &info, sizeof(AVISTREAMINFO)))
			throw("Error while obtaining stream info.");
		framerate = (float)info.dwRate / (float)info.dwScale;
		if (framerate <= 0.0) framerate = 24.0f;

		LONG lSize; // in bytes
		if (AVIStreamReadFormat(pVideo[nStream], AVIStreamStart(pVideo[nStream]), NULL, &lSize))
		   throw("Error while determining stream size of the video file.");
		
		pChunk = (LPBITMAPINFO) new BYTE[lSize];
		if (!pChunk) throw("Error while allocating memory for the video file.");

		if (AVIStreamReadFormat(pVideo[nStream], AVIStreamStart(pVideo[nStream]), pChunk, &lSize))
			throw("Error while reading stream format of the video file.");

		// Specify the format we want our images in
		BITMAPINFOHEADER wantedFormat;
		memcpy(&wantedFormat, &(pChunk->bmiHeader), sizeof(BITMAPINFOHEADER));
		wantedFormat.biBitCount = 24;
		wantedFormat.biCompression = BI_RGB;
		wantedFormat.biSizeImage = 0;
		frameExtent = make_uint2(wantedFormat.biWidth, wantedFormat.biHeight);

		// Open the stream
		pgf = AVIStreamGetFrameOpen(pVideo[nStream], &wantedFormat);
		if (!pgf) throw("Error while opening a frame of the video file.");

		// Allocate memory for the volume
		endFrame = AVIStreamEnd(pVideo[nStream]);
	}
}

unsigned char* GetFrame(LONG lFrame, uint2& frameExtent)
{
	// Pluck a packed DIB from the video stream
	LPBITMAPINFOHEADER lpbi = (LPBITMAPINFOHEADER)AVIStreamGetFrame(pgf, lFrame);
	frameExtent = make_uint2(lpbi->biWidth, lpbi->biHeight);
	unsigned char* pData = (unsigned char*)lpbi + lpbi->biSize;
	//const int frameSize = lpbi->biWidth * lpbi->biHeight;
	return pData;
}

void computeFPS()
{
	const char* method[] = {"Nearest neighbor interpolation", "Linear interpolation", "Fast cubic interpolation", "Non-prefiltered fast cubic weighting"};
	const float updateInterval = 1000.0f/6.0f;
	static float timeStamp = 0.0f;
	static int counter = 0;
	char str[256];

	counter++;
	float now = cutGetTimerValue(hTimer);
	if (!animate)
	{
		sprintf(str, "%s", method[filterMethod]);
		glutSetWindowTitle(str);
	}
	else if (now-timeStamp > updateInterval)
	{
		float framerate = 1000.0f * (float)counter / (now-timeStamp);
		sprintf(str, "%s, Framerate: %3.1f fps", method[filterMethod], framerate);
		glutSetWindowTitle(str);
		counter = 0;
		timeStamp = now;
	}
}

// display results using OpenGL (called by GLUT)
void display()
{
	static float timeStamp = -2000.0f;
	float now = cutGetTimerValue(hTimer);
	if ((now-timeStamp)*framerate > 1000.0f)
	{
		// Get a new frame
		static LONG index = 0;
		if (animate) index = (LONG)(0.001f * (now-timeStamp) * framerate + 0.5f) % endFrame;
		uchar* rgb = GetFrame(index, frameExtent);
		initTexture(rgb, frameExtent.x, frameExtent.y, filterMethod);
		if (timeStamp < 0.0f) timeStamp = now;
	}

	// map PBO to get CUDA device pointer
    uchar4* output;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&output, pbo));
    render(output, windowExtent, frameExtent, filterMethod);  //call the render routine in _kernel.cu
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbo));

#ifndef _NO_DISPLAY
    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(windowExtent.x, windowExtent.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();
    glutReportErrors();
#endif

	computeFPS();
}

void idle()
{
    if (animate) glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch(key) {
        case 27:
            exit(0);
            break;
		case 'f':
			filterMethod = (filterMethod + 1) % 4;
			break;
        case ' ':
            animate = !animate;
            break;
        default:
            break;
    }
    glutPostRedisplay();
}

void initOpenGL()
{
	#if defined(_WIN32)
	//if (wglSwapIntervalEXT) wglSwapIntervalEXT(GL_FALSE);  //disable vertical synchronization
	#endif

	if (pbo != 0)
	{
		CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));    
		glDeleteBuffersARB(1, &pbo);
	}

    // create pixel buffer object
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, 4*windowExtent.x*windowExtent.y*sizeof(GLubyte), 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbo));
}

void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
	windowExtent = make_uint2(x, y);
	initOpenGL();
}

int usage(const char* program)
{
	printf("Usage: %s filename\n", program);
	printf("\tfilename: name of the file containing the AVI movie\n");
	return -1;
}

void cleanup()
{
	CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));    
	glDeleteBuffersARB(1, &pbo);

	// release the AVI engine
	delete[] pChunk;
	if (AVIStreamGetFrameClose(pgf)) throw("Error while closing a frame of the video file.");
	AVIFileRelease(pAviFile);  //closes the file 
    AVIFileExit();  //releases AVIFile library
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	if (argc < 2) return usage(argv[0]);
	CUT_DEVICE_INIT(argc-1, argv);
	CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	CUT_SAFE_CALL(cutStartTimer(hTimer));

	// Open the AVI file
    try
    {
	    LoadAVI(argv[1]);
    }
    catch (char* msg)
    {
        MessageBox(NULL, (std::string(msg) + "\nCheck whether the video codec is installed.").c_str(), "Error", 0);
        return -1;
    }

	double scale = 1024.0 / frameExtent.x;
	windowExtent = make_uint2(scale * frameExtent.x, scale * frameExtent.y);

    printf("Press space to toggle animation\n"
           "Press 'f' to toggle between nearest neighbour, linear, simple cubic and\n"
           "fast cubic texture filtering\n"
           "Press '+' and '-' to change displayed slice\n");

    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(windowExtent.x, windowExtent.y);
	glutCreateWindow("CUDA cubic AVI playback");
	//glutFullScreen();
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }
    initOpenGL();

    atexit(cleanup);

    glutMainLoop();
    return 0;
}
