#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "OpenGLInstanceViewer.h"
#include <cuda_gl_interop.h>



#define GRID_WIDTH 128			/** Problem size width */
#define GRID_HEIGHT 128			/** Problem size height */

struct cudaGraphicsResource *cuda_tbo_resource;		/** CUDA graphics resource is registered with the texture buffer object */

float time_interval = 0.0;							/** Time interval used for advancing the simulation */

void setVertexData(float* verts);
void setVertexInstanceData(unsigned int *vertex_instance_ids);
void executeSimulation();

// Vertex shader source code
const char vertexShaderSource[] =
{
	"#extension GL_EXT_gpu_shader4 : enable												\n"
	"uniform samplerBuffer instance_tex;												\n"
	"attribute in uint instance_index;													\n"
	"void main()																		\n"
	"{																					\n"
	"	vec4 instance_data = texelFetchBuffer(instance_tex, (int)instance_index);		\n"
	"	vec4 position = vec4(gl_Vertex.x, gl_Vertex.y, instance_data.w, 1.0f);			\n"
	"	gl_FrontColor = vec4(instance_data.x, instance_data.y, instance_data.z, 0.0f);	\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    				\n"
	"}																					\n"
};

__global__ void simple_instance_kernel(float4 *instance_data, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;


	// write data to instance data (mapped form TBO)
	float red = x / (float)width;
	float green = y / (float)height;
	float blue = 0.0f;

	// Exercise 1.6) Create a displacement value
	float freq = 4.0f;
	float H = 0.1f;
	float displacement = sinf(red*freq + time) * cosf(green*freq + time) * H;

	instance_data[y*width + x] = make_float4(red, green, blue, displacement);
}


void executeSimulation()
{
	size_t num_bytes;
	float4 *dptr;

	// Exercise 1.1) Map CUDA graphics resource
	cudaGraphicsMapResources(1, &cuda_tbo_resource);

	// Exercise 1.2) Map the TBO buffer to device pointer and check size
	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_tbo_resource);
	if (num_bytes != GRID_WIDTH*GRID_HEIGHT * 4 * sizeof(float)){
		printf("Warning: CUDA mapped pointer has unexpected size!\n");
	}

	// Exercise 1.3) Call the kernel
	dim3 block(8, 8, 1);
	dim3 grid(GRID_WIDTH / block.x, GRID_HEIGHT / block.y, 1);
	simple_instance_kernel << <grid, block >> >(dptr, GRID_WIDTH, GRID_HEIGHT, time_interval);

	// Exercise 1.4) Unmap the CUDA graphics resource
	cudaGraphicsUnmapResources(1, &cuda_tbo_resource);

	//increment the time interval
	time_interval += 0.01f;
}


int main(int argc, char* argv[])
{
	//We need to set the CUDA device and CUDA GL device before initialising any OpenGL
	cudaSetDevice(0);
	cudaGLSetGLDevice(0);

	// Initialise the instance viewer there are GRID_WIDTH*GRID_HEIGHT instances and 4 vertices per instance (GL_QUADS)
	initInstanceViewer(GRID_WIDTH * GRID_HEIGHT, 4);

	// Initialise the vertex shader used for instancing the vertex data
	initInstanceShader(vertexShaderSource);

	// Initialise the vertex data and instance data
	initVertexData(setVertexData, setVertexInstanceData);

	// Register the instance data Texture Buffer Object (TBO) with CUDA
	cudaGraphicsGLRegisterBuffer(&cuda_tbo_resource, getInstanceTBO(), cudaGraphicsMapFlagsWriteDiscard);

	// Set the pre render function to execute the CUDA kernel and update the TBO
	setPreRenderFunction(executeSimulation);

	// Enter the main render loop (blocking function)
	beginRenderLoop();

	//un-register the CUDA TBO
	cudaGraphicsUnregisterResource(cuda_tbo_resource);

	// On return from the render loop clean-up
	cleanupInstanceViewer();

}

void setVertexData(float* verts)
{
	float quad_width;
	float quad_height;

	//all veterx data is to be normalised between -0.5 and +0.5 in x and y dimensions
	quad_width = 1.0f / (float)(GRID_WIDTH);
	quad_height = 1.0f / (float)(GRID_HEIGHT);

	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			int offset = (x + (y * (GRID_WIDTH))) * 3 * 4;

			float x_min = (float)x / (float)(GRID_WIDTH);
			float y_min = (float)y / (float)(GRID_HEIGHT);

			//first vertex
			verts[offset + 0] = x_min - 0.5f;
			verts[offset + 1] = y_min - 0.5f;
			verts[offset + 2] = 0.0f;

			//second vertex
			verts[offset + 3] = x_min - 0.5f;
			verts[offset + 4] = y_min + quad_height - 0.5f;
			verts[offset + 5] = 0.0f;

			//third vertex
			verts[offset + 6] = x_min + quad_width - 0.5f;
			verts[offset + 7] = y_min + quad_height - 0.5f;
			verts[offset + 8] = 0.0f;

			//fourth vertex
			verts[offset + 9] = x_min + quad_width - 0.5f;
			verts[offset + 10] = y_min - 0.5f;
			verts[offset + 11] = 0.0f;
		}
	}
}

void setVertexInstanceData(unsigned int *vertex_instance_ids)
{
	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			int index = (x + (y * (GRID_WIDTH)));
			int offset = index * 4;

			//four vertices (a quad) have the same instance index
			vertex_instance_ids[offset + 0] = index;
			vertex_instance_ids[offset + 1] = index;
			vertex_instance_ids[offset + 2] = index;
			vertex_instance_ids[offset + 3] = index;
		}
	}
}
