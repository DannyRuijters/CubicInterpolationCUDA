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

#include "tricubic.shader"
#include <stdlib.h>
#include <stdio.h>
#include <GL/glew.h>
#if defined(_WIN32)
#include <GL/wglew.h>
#endif

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#ifndef max
#define max(a,b) ((a>b)?a:b)
#endif

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned char voxeltype;

uint width = 512, height = 512;  //output window
static struct {uint x,y,z;} volumeSize;
voxeltype* voxels = NULL;
float xAngle = 0.0f, yAngle = 0.0f;
float xTranslate = 0.0f, yTranslate = 0.0f;
float zoom = 0.75f;
enum enMouseButton {mouseLeft, mouseMiddle, mouseRight, mouseNone} mouseButton = mouseNone;
int mouseX = 0, mouseY = 0;
float tfloor=0.0f, tceil=1.0f;

GLuint renderBufferObj = 0;
GLuint texBack = 0;
GLuint texVoxels = 0;
GLuint fbo = 0;
GLuint programRayCast = 0;
uint filterMethod = 1;


void computeFPS()
{
	const char* method[] = {"Linear interpolation", "Fast cubic weighting"}; //"Nearest neighbor"
	const float updatesPerSec = 6.0f;
	static int counter = 0;
	static int countLimit = 0;
	static int last = glutGet(GLUT_ELAPSED_TIME);

	if (counter++ > countLimit)
	{
		int now = glutGet(GLUT_ELAPSED_TIME);
		float framerate = 1000.0f * (float)counter / (float)(now - last);
		char str[256];
		sprintf(str, "%s, Framerate: %3.1f fps", method[filterMethod], framerate);
		glutSetWindowTitle(str);
		countLimit = (int)(framerate / updatesPerSec);
		counter = 0;
		last = now;
	}
}

void Load3DTexture()
{
	glBindTexture(GL_TEXTURE_3D, texVoxels);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_INTENSITY, volumeSize.x, volumeSize.y, volumeSize.z, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, voxels);

	glUseProgram(programRayCast);
	GLint location = glGetUniformLocation(programRayCast, "sampleInterval");
	glUniform3f(location, 0.5 /(volumeSize.x-1), 0.5 /(volumeSize.y-1), 0.5 /(volumeSize.z-1));
	glUseProgram(0);
}

void drawTextureCoords()
{
	glPushMatrix();
	double vmax = max(max(volumeSize.x, volumeSize.y), volumeSize.z) - 1.0;
	glScaled((volumeSize.x-1)/vmax, (volumeSize.y-1)/vmax, (volumeSize.z-1)/vmax);
	const double lo[3] = {0.5/(volumeSize.x-1), 0.5/(volumeSize.y-1), 0.5/(volumeSize.z-1)};
	const double hi[3] = {1.0-lo[0], 1.0-lo[1], 1.0-lo[2]};

	glBegin(GL_QUADS);

	for (int side = 0; side <= 1; side++)
	{
		const int move = 2 * side - 1;
		const double* loSide = (side == 0) ? lo : hi;
		const double* hiSide = (side == 0) ? hi : lo;
		glColor3d(lo[0], loSide[1], loSide[2]); glVertex3f(-1,  move, move);
		glColor3d(lo[0], hiSide[1], loSide[2]); glVertex3f(-1, -move, move);
		glColor3d(hi[0], hiSide[1], loSide[2]); glVertex3f( 1, -move, move);
		glColor3d(hi[0], loSide[1], loSide[2]); glVertex3f( 1,  move, move);
		
		glColor3d(lo[0], loSide[1], hiSide[2]); glVertex3f(-1, move, -move);
		glColor3d(lo[0], loSide[1], loSide[2]); glVertex3f(-1, move,  move);
		glColor3d(hi[0], loSide[1], loSide[2]); glVertex3f( 1, move,  move);
		glColor3d(hi[0], loSide[1], hiSide[2]); glVertex3f( 1, move, -move);

		glColor3d(loSide[0], lo[1], loSide[2]); glVertex3f(move, -1,  move);
		glColor3d(loSide[0], lo[1], hiSide[2]); glVertex3f(move, -1, -move);
		glColor3d(loSide[0], hi[1], hiSide[2]); glVertex3f(move,  1, -move);
		glColor3d(loSide[0], hi[1], loSide[2]); glVertex3f(move,  1,  move);
	}

	glEnd();
	glPopMatrix();
}

// display results using OpenGL (called by GLUT)
void display()
{
//Load3DTexture();

	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glPushMatrix();
	glTranslatef(xTranslate, yTranslate, 0.0f);
	glScalef(zoom, zoom, 0.1f);
	glRotatef(xAngle, 0, 1, 0);
	glRotatef(yAngle, 1, 0, 0);

	// Draw the texture coordinates for the rays
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderBufferObj);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texBack, 0);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	glEnable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glCullFace(GL_FRONT);
	drawTextureCoords();
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glEnable(GL_BLEND);
	
	// Bind textures
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, texBack);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_3D, texVoxels);
	glActiveTexture(GL_TEXTURE0);
	
	// Use the program object
	glUseProgram(programRayCast);
	GLint location = glGetUniformLocation(programRayCast, "slopeIntercept");
	glUniform2f(location, 1.0f/(tceil-tfloor), -tfloor/(tceil-tfloor));
	location = glGetUniformLocation(programRayCast, "method");
	glUniform1i(location, filterMethod);

	glCullFace(GL_BACK);
	drawTextureCoords();
	glDisable(GL_CULL_FACE);
	glUseProgram(0);
	glPopMatrix();

	computeFPS();
    glutSwapBuffers();
    glutReportErrors();
}

void idle()
{
	glutPostRedisplay();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
	{
    case 27:
        ::exit(0);
        break;
	case 'f':
		filterMethod = (filterMethod + 1) % 2;
		break;
	case ',':
		tfloor -= 0.01f;
		tceil -= 0.01f;
		break;
	case '.':
		tfloor += 0.01f;
		tceil += 0.01f;
		break;
	case '[':
		tceil += 0.01f;
		break;
	case ']':
		tceil -= 0.01f;
		break;
    default:
        break;
    }
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) mouseButton = (enMouseButton)button;
	else mouseButton = mouseNone;

	mouseX = x;
	mouseY = y;
}

void motion(int x, int y)
{
	switch (mouseButton)
	{
	case mouseLeft:
		xAngle += x - mouseX;
		yAngle += y - mouseY;
		glutPostRedisplay();
		break;
	case mouseMiddle:
		xTranslate += 0.005f * (x - mouseX);
		yTranslate -= 0.005f * (y - mouseY);
		break;
	case mouseRight:
		zoom += 0.01f * (y - mouseY);
		glutPostRedisplay();
		break;
	case mouseNone:
	default:
		//nothing
		break;
	}

	mouseX = x;
	mouseY = y;
}


const char raycast[] =
"uniform sampler2D texBack;\n"
"uniform sampler3D voxels;\n"
"uniform int method;\n"
"uniform vec3 sampleInterval;\n"
"uniform vec2 slopeIntercept;\n"
"varying vec2 texPos;\n"

"float interpolate_tricubic_fast(sampler3D tex, vec3 coord);\n"

"void main()\n"
"{\n"
"	vec3 start = gl_Color.xyz;\n"
"	vec3 end = texture2D(texBack, texPos).xyz;\n"
"	vec3 dir = end - start;\n"
"	float rayLength = length(dir);\n"

"	vec3 rayColor = vec3(0.0);\n"
"	float rayAlpha = 0.0;\n"

"	if (rayLength > 0.1)\n"
"	{\n"
"		const vec3 frontClr = vec3(1.0, 0.9, 0.5);\n"
"		const vec3 backClr = vec3(0.5, 0.5, 1.0);\n"
"		const float alphaThreshold = 0.95;  //opacity threshold for early ray termination of saturated rays\n"
"		float sampleDistance = dot(sampleInterval, abs(dir)) / (rayLength * rayLength);  //distance between the samples\n"

"		for (float t = 0.0; t < 1.0 && rayAlpha < alphaThreshold; t += sampleDistance)\n"
"		{\n"
"			vec3 pos = start + t * dir;\n"
"			float sample = (method==0) ? texture3D(voxels, pos).x : interpolate_tricubic_fast(voxels, pos);\n"
"			sample = slopeIntercept.x * sample + slopeIntercept.y;\n"
"			sample = clamp(sample, 0.0, 1.0);\n"
"			vec3 color = sample * mix(frontClr, backClr, t);\n"
"			vec4 lookup = vec4(color, sample);\n"

			// Under operator
"			rayColor += (1.0 - rayAlpha) * lookup.a * lookup.rgb;\n"
"			rayAlpha += (1.0 - rayAlpha) * lookup.a;\n"
"		}\n"
"	}\n"

// write output color
"	gl_FragColor = vec4(rayColor, rayAlpha);\n"
"}\n";

// Create a shader object, load the shader source, and compile the shader.
GLuint LoadShader(GLenum type, const char *shaderSrc)
{
	// Create the shader object
	GLuint shader = glCreateShader(type);
	if (shader == 0) return 0;
	// Load the shader source
	glShaderSource(shader, 1, &shaderSrc, NULL);
	// Compile the shader
	glCompileShader(shader);
	// Check the compile status
	GLint compiled;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
	if (compiled == 0)
	{
		GLint infoLen = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
		if (infoLen > 1)
		{
			char* infoLog = new char[infoLen];
			glGetShaderInfoLog(shader, infoLen, NULL, infoLog);
			fprintf(stderr, "Error compiling shader:\n%s\n", infoLog);
			delete[] infoLog;
		}
		glDeleteShader(shader);
		return 0;
	}

	glutReportErrors();
	return shader;
}

bool LinkProgram(GLuint programObject)
{
	// Link the program
	glLinkProgram(programObject);
	// Check the link status
	GLint linked;
	glGetProgramiv(programObject, GL_LINK_STATUS, &linked);
	if (linked == 0)
	{
		GLint infoLen = 0;
		glGetProgramiv(programObject, GL_INFO_LOG_LENGTH, &infoLen);
		if (infoLen > 1)
		{
			char* infoLog = new char[infoLen];
			glGetProgramInfoLog(programObject, infoLen, NULL, infoLog);
			fprintf(stderr, "Error compiling shader:\n%s\n", infoLog);
			delete[] infoLog;
		}
		return false;
	}

	return true;
}

// Initialize the shader and program object
void initShader()
{
	char vShaderStr[] =
		"varying vec2 texPos;\n"
		"void main()\n"
		"{\n"
		"	gl_FrontColor = gl_Color;\n"
		"   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
		"	texPos = vec2(0.5 * gl_Position + 0.5);\n"
		"}\n";

	// Load the vertex/fragment shaders
	GLuint vertexShader = LoadShader(GL_VERTEX_SHADER, vShaderStr);
	GLuint fragmentShader = LoadShader(GL_FRAGMENT_SHADER, raycast);
	GLuint cubicFragmentShader = LoadShader(GL_FRAGMENT_SHADER, tricubic);
	// Create the program object
	programRayCast = glCreateProgram();
	glAttachShader(programRayCast, vertexShader);
	glAttachShader(programRayCast, fragmentShader);
	glAttachShader(programRayCast, cubicFragmentShader);
	
	if (LinkProgram(programRayCast))
	{
		glUseProgram(programRayCast);
		// Bind the uniform samplers
		GLint location = glGetUniformLocation(programRayCast, "texBack");
		glUniform1i(location, 1);
		location = glGetUniformLocation(programRayCast, "voxels");
		glUniform1i(location, 2);
		glUseProgram(0);
	}

	glutReportErrors();
}

void initOpenGL()
{
	#if defined(_WIN32)
	//if (wglSwapIntervalEXT) wglSwapIntervalEXT(GL_FALSE);  //disable vertical synchronization
	#endif

	// create framebuffer and renderbuffer
	glGenTextures(1, &texBack);
	glBindTexture(GL_TEXTURE_2D, texBack);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glGenRenderbuffersEXT(1, &renderBufferObj);
	glGenFramebuffersEXT(1, &fbo);
	glGenTextures(1, &texVoxels);

	glutReportErrors();
}

void cleanup()
{
	if (fbo) glDeleteFramebuffersEXT(1, &fbo);
	if (renderBufferObj) glDeleteRenderbuffersEXT(1, &renderBufferObj);
	if (texBack) glDeleteTextures(1, &texBack);
	if (texVoxels) glDeleteTextures(1, &texVoxels);
	if (programRayCast) glDeleteProgram(programRayCast);
	
	fbo = 0;
	texBack = 0;
	texVoxels = 0;
	programRayCast = 0;
}

void resize(int w, int h)
{
	width = w;
	height = h;
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glScaled(1.0 , (double)width/(double)height, 1.0);
	glMatrixMode(GL_MODELVIEW);

	// (Re)create the textures and buffers
	glBindTexture(GL_TEXTURE_2D, texBack);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, width, height, 0, GL_RGB, GL_FLOAT, 0);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderBufferObj);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, width, height);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
}

int usage(const char* program)
{
	printf("Usage: %s filename voxelsX voxelsY voxelsZ\n", program);
	printf("\tfilename: name of the file containing the raw 8-bit voxel data\n");
	printf("\tvoxelsX: number of voxels in x-direction\n");
	printf("\tvoxelsY: number of voxels in y-direction\n");
	printf("\tvoxelsZ: number of voxels in z-direction\n");
	printf("\texample: %s ../data/bucky.raw 32 32 32\n\n", program);
	return -1;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	if (argc < 5) return usage(argv[0]);

	// Obtain the voxel data
	volumeSize.x = atoi(argv[2]);
	volumeSize.y = atoi(argv[3]);
	volumeSize.z = atoi(argv[4]);
	if (volumeSize.x <= 0 || volumeSize.y <= 0 || volumeSize.z <= 0) return usage(argv[0]);
	
	FILE* fp = fopen(argv[1], "rb");
	if (fp == NULL) {
		printf("Could not open file %s\n", argv[1]);
		return usage(argv[0]);
	}

	size_t nrOfVoxels = volumeSize.x * volumeSize.y * volumeSize.z;
	voxels = new voxeltype[nrOfVoxels];
	size_t linesRead = fread(voxels, volumeSize.x * sizeof(voxeltype), volumeSize.y * volumeSize.z, fp);
	fclose(fp);
	if (linesRead * volumeSize.x != nrOfVoxels) {
		delete[] voxels;
		printf("Error: The number of voxels read does not correspond to the number specified!\n");
		return usage(argv[0]);
	}

	printf("\nPress 'f' to change the interpolation mode.\n");
	printf("Use the mouse to rotate, translate and scale the volume.\n");

    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("OpenGL 3D texture");
	glutReshapeFunc(resize);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
    glutIdleFunc(idle);

    glewInit();
    initOpenGL();
	initShader();
	Load3DTexture();

    atexit(cleanup);

    glutMainLoop();
    return 0;
}
