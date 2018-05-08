/**
* @file OpenGLInstanceViewer.h
* @author Paul Richmond
* @brief Simple OpenGL module for creating visualisation of instanced data which can be updated externally via CUDA
*
* File is part of Lab 09 from http://paulrichmond.shef.ac.uk/teaching/COM4521/
*/

//Header guards prevent the contents of the header from being defined multiple times where there are circular dependencies
#ifndef __VIEWER_HEADER__
#define __VIEWER_HEADER__


// OpenGL Graphics includes
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/glew.h>
#include <GL/glut.h>

#define REFRESH_DELAY 10		/** Refresh delay controls how frequently the scene should be re-drawn (measured in ms) */
#define WINDOW_WIDTH 1024		/** Window width */
#define WINDOW_HEIGHT 768		/** Window height */

// C++ guard is required as NVCC actually produced C++ objects. For a C++ object to link with a C function it must use the extern "C" declaration. This is extern "C" is ignored by the C file which implements these functions.
// An alternative to the guarded syntax would be to compile the source module OpenGLInstanceViewer.c as CUDA source with nvcc (jn which case the extern is not required)
#ifdef __cplusplus
extern "C" {
#endif

	/**
	 * @brief Initialisation function for the Viewer. Must be called first.
	 *
	 * This function must be called first or errors will be raised. The function will take care of initialising OpenGL context and the GLUT rendering window. Default mouse and keyboard handles will be set. These can be overwritten by setting your own handle functions with glutDisplayFunc() etc.
	 *
	 * @param instances The number of instances for rendering.
	 * @param num_vertices_per_instance The fixed number of vertices per instance. If this is not 4 (i.e. GL_QUAD) then setRenderMode() function must be used to change how the vertices should be interpreted.
	 */
	void initInstanceViewer(int instances, int num_vertices_per_instance);

	/**
	* @brief Initialises the user defined OpenGL Shader (GLSL) for instance rending.
	*
	* Shader must contain 'attribute in uint instance_index' This associates each vertex with an unique instance. You can use this instance data to change your vertex data (e.g. by displacing it or manipulating the vertex colours). There is no restriction on the name of the 'uniform sampleBuffer' which you use within your shader to access instance data. You can fill this sample buffer with data by accessing its Texture Buffer Object (TBO) GL handle via the getInstanceTBO() function and mapping it CUDA memory.
	*
	* @param shader_src The shader program for instancing. Compile errors will be reported at runtime in the console.
	*/
	void initInstanceShader(const char* shader_src);

	/**
	* @brief Function for setting a function handler to initialise vertices and vertex instance identifiers
	*
	* This function must receive pointers to functions to be used for initialising the vertex data and vertices instance ids (i.e. which instance each vertex belongs to). The function takes care of generating vertex buffer and attribute objects as well and performing binding and unbinding of buffers so that your functions can fill the buffers with data.
	*
	* @param setVertexDataFunc A function pointer to a function which accepts a pointer to float. The function argument will be mapped to memory with available size dictated by to the number of instances and the number of vertices per instance which were specified on initialisation. For each unique vertex there are three float values (for x, y and z positions). These are laid out consecutively in memory. Reading beyond the total number of floats will result in a memory exception. 
	* @param setVertexInstanceDataFunc A function pointer to a function which accepts a pointer to unsigned int. The function argument will be mapped to memory with available size dictated by the number of instances and the number of vertices per instance which were specified on initialisation. For each unique vertex there is a single unsigned int value for labelling the instance identifier. Reading beyond the total number of unsigned ints will result in a memory exception. 
	*/
	void initVertexData(void(*setVertexDataFunc)(float* verts), void(*setVertexInstanceDataFunc)(unsigned int* vertex_instance_ids));
	
	/**
	* @brief Function for setting a function handler to perform updates to the Instanced data Texture Buffer Object returned by getInstanceTBO()
	*
	* This function sets the callback function to be used for updating instance data before rending. Typically this will be done by binding the Instance TBO to CUDA memory.
	*
	* @param preDisplay A function pointer to a function which updates the Instanced data Texture Buffer Object (TBO) by mapping it to CUDA memory
	*/
	void setPreRenderFunction(void(*preDisplay)());
	
	/**
	* @brief Begins the rendering loop
	*
	* This function enters the main rendering loop and will not exit until the program is closed (via window handle or q key) or an exit call is made.
	*/
	void beginRenderLoop();

	/**
	* @brief Cleanup function
	*
	* This function unbinds any buffers and deallocates any memory
	*/
	void cleanupInstanceViewer();


	/**
	* @brief Returns a handle to the Instanced data Texture Buffer Object (TBO)
	*
	* The Texture Buffer Object should be updated by binding to CUDA memory and performing some operation to set the 4 available float values per instance. The TBO is automatically bound for rendering and can be sampled in the instance shared using the unique vertex instance id passed as a vertex attribute.
	*
	* @return TBO handle
	*/
	GLuint getInstanceTBO();


	/**
	* @brief Function for setting the render mode from anything but GL_QUADS
	*
	* If your instanced data does not have 4 vertices per instance (i.e. not quad geometry) then the render mode should be set to reflect how the vertex data should be interpreted.
	*
	* @return TBO handle
	*/
	void setRenderMode(GLenum render_mode);


#ifdef __cplusplus
}
#endif

#endif //__VIEWER_HEADER__