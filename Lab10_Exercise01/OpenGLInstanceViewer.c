#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "OpenGLInstanceViewer.h"


// global variables
unsigned int numInstances = 0;
unsigned int vertsPerInstance = 4;
GLenum renderMode = GL_QUADS;

// buffers and texture handles
GLuint vao = 0;
GLuint vao_vertices = 0;
GLuint vao_instance_ids = 0;
GLuint tbo = 0;
GLuint tex = 0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_z = 0.0;
float translate_z = -1.0;

// vertex shader handles
GLuint vs_shader = 0;
GLuint vs_program = 0;
GLuint vs_instance_index = 0;

// function pointer to function to execute before rending
void(*preDisplay)();

// function prototypes
void initVertexShader();
void cleanup();
void timerRefresh(int value);
void displayLoop(void);
void initGL();
void checkGLError();
void noPreRenderFunc();
void handleKeyboardDefault(unsigned char key, int x, int y);
void handleMouseDefault(int button, int state, int x, int y);
void handleMouseMotionDefault(int x, int y);

/* Public function definitions (defined in header) */

void initInstanceViewer(int instances, int num_vertices_per_instance)
{
	initGL();
	numInstances = instances;
	vertsPerInstance = num_vertices_per_instance;
	preDisplay = noPreRenderFunc;
}

void initInstanceShader(const char* shader_src)
{
	if (numInstances == 0){
		printf("Error: initInstanceViewer must be called before initInstanceShader\n");
		return;
	}

	//vertex shader
	vs_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs_shader, 1, &shader_src, 0);
	glCompileShader(vs_shader);


	// check for errors
	GLint status;
	glGetShaderiv(vs_shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[1024];
		int len;
		glGetShaderInfoLog(vs_shader, 1024, &len, data);
		printf("%s", data);
	}
	
	
	//program
	vs_program = glCreateProgram();
	glAttachShader(vs_program, vs_shader);
	glLinkProgram(vs_program);
	glGetProgramiv(vs_program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Program Link Error\n");
	}

	// get shader variables
	vs_instance_index = glGetAttribLocation(vs_program, "instance_index");
	if (vs_instance_index == 0){
		printf("Warning: Shader program missing 'attribute in uint instance_index'\n");
	}

	glUseProgram(0);
	//check for any errors
	checkGLError();
}

void initVertexData(void(*setVertexDataFunc)(float* verts), void(*setVertexInstanceDataFunc)(unsigned int* vertex_instance_ids))
{
	if (vertsPerInstance != 4 && renderMode == GL_QUADS)
		printf("Warning: Number of vertices per instance is not 4 but render mode is Quads. Try changing the render mode\n");

	if (vs_shader == 0){
		printf("Error: initInstanceShader must be called before initVertexData\n");
		return;
	}

	/* vertex array object */
	glGenVertexArrays(1, &vao); // Create our Vertex Array Object  
	glBindVertexArray(vao); // Bind our Vertex Array Object so we can use it  

	/* create a vertex buffer */

	// create buffer object
	glGenBuffers(1, &vao_vertices);
	glBindBuffer(GL_ARRAY_BUFFER, vao_vertices);
	glBufferData(GL_ARRAY_BUFFER, numInstances * vertsPerInstance * 3 * sizeof(float), 0, GL_STATIC_DRAW);
	float* verts = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	setVertexDataFunc(verts);
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, 0); // Set up our vertex attributes pointer 
	glEnableVertexAttribArray(0);
	checkGLError();

	// instance index buffer
	glGenBuffers(1, &vao_instance_ids);
	glBindBuffer(GL_ARRAY_BUFFER, vao_instance_ids);
	glBufferData(GL_ARRAY_BUFFER, numInstances * vertsPerInstance * sizeof(unsigned int), 0, GL_STATIC_DRAW);
	unsigned int* ids = (unsigned int*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	setVertexInstanceDataFunc(ids);
	if (vs_shader == 0){
		printf("Warning: Vertex shader must be initialised before buffers\n");
	}
	else{
		glVertexAttribIPointer((GLuint)vs_instance_index, 1, GL_UNSIGNED_INT, 0, 0); // Set up instance id attributes pointer in shader
		glEnableVertexAttribArray(vs_instance_index);
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);


	checkGLError();

	/* texture buffer object */

	glGenBuffers(1, &tbo);
	glBindBuffer(GL_TEXTURE_BUFFER_EXT, tbo);
	glBufferData(GL_TEXTURE_BUFFER_EXT, numInstances * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);		// 4 float elements in a texture buffer object per instance for rbga data

	/* generate texture */
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, tex);
	glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F, tbo);

	//unbind buffers
	glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);

	checkGLError();
}

void beginRenderLoop()
{
	glutMainLoop();
}

void cleanupInstanceViewer()
{
	glBindVertexArray(vao);

	glDeleteBuffers(1, &vao_vertices);
	vao_vertices = 0;

	glDeleteBuffers(1, &vao_instance_ids);
	vao_instance_ids = 0;

	glDeleteBuffers(1, &tbo);
	tbo = 0;

	glDeleteTextures(1, &tex);
	tex = 0;

	glDeleteVertexArrays(1, &vao);
	vao = 0;

	checkGLError();
}


GLuint getInstanceTBO()
{
	return tbo;
}

void setPreRenderFunction(void(*preDisplayFunc)()){
	preDisplay = preDisplayFunc;
}


void setRenderMode(GLenum render_mode)
{
	renderMode = render_mode;
}
/* Private functions */


void initGL()
{
	int argc = 1;
	char * argv[] = { "Com4521 - OpenGL Instance Viewer" };

	//glut init
	glutInit(&argc, argv);

	//init window
	glutInitDisplayMode(GLUT_RGB);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Com4521 - OpenGL Instance Viewer");

	// glew init (must be done after window creation for some odd reason)
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 "))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		exit(0);
	}

	// register default callbacks
	glutDisplayFunc(displayLoop);
	glutKeyboardFunc(handleKeyboardDefault);
	glutMotionFunc(handleMouseMotionDefault);
	glutMouseFunc(handleMouseDefault);
	glutTimerFunc(REFRESH_DELAY, timerRefresh, 0);


	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)WINDOW_WIDTH / (GLfloat)WINDOW_HEIGHT, 0.001, 10.0);
}


void timerRefresh(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerRefresh, 0);
	}
}

void displayLoop(void)
{
	//call user defined pre display function
	preDisplay();

	// set view matrix and prepare for rending
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//transformations
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_z, 0.0, 0.0, 1.0);

	// attach the shader program to rendering pipeline to perform per vertex instance manipulation 
	glUseProgram(vs_program);

	// Bind our Vertex Array Object  (contains vertex buffers object and vertex attribute array)
	glBindVertexArray(vao); 

	// Bind and activate texture with instance data (held with the TBO)
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, tex);

	// Draw the vertices with attached vertex attribute pointers
	glDrawArrays(renderMode, 0, vertsPerInstance * numInstances); 

	// Disable the shader program and return to the fixed function pipeline
	glUseProgram(0);

	glutSwapBuffers();
}


void checkGLError(){
	int Error;
	if ((Error = glGetError()) != GL_NO_ERROR)
	{
		const char* Message = (const char*)gluErrorString(Error);
		fprintf(stderr, "OpenGL Error : %s\n", Message);
	}
}

void noPreRenderFunc()
{
	//Empty function. Default function if no pre render function is set.
}


void handleKeyboardDefault(unsigned char key, int x, int y)
{
	switch (key) {
	case(27) : //escape key
		//return control to the users program to allow them to clean-up any allcoated memory etc.
		glutLeaveMainLoop();
		break;
	}
}

void handleMouseDefault(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void handleMouseMotionDefault(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_z += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}