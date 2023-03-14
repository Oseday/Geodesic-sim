#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <vector_types.h>

// Custom vector operations & functions
#include "vector_helper.cuh"

// Device math functions
//#include <crt/math_functions.h>

// Device sorting
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// display defs
#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

// Making <<< >>> intellisense compatible
#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

// Syncing CUDA kernels
#define CUDASYNC checkCudaErrors(cudaDeviceSynchronize());

#define DO_COLLISIONS
//#define NAIVE_GRAVITY
//#define NAIVE_COLLISIONS 

// constants
unsigned int window_width  = 512;
unsigned int window_height = 512;

const int BlockSize = 128;//350;//112;
const int ThreadSize = 512;//512;

const int NBodyCount = BlockSize * ThreadSize;

#define Collision_Elasticity 0.8f
const int Collision_Iterations = 2;
#define minDist 0.035f //0.008f
#define Boundary_Distance 10.0f


const float Gravitational_Constant = 5.0;
const float Center_Pull = 0.000f;

const float Omega_Rot_Speed = 140;

float Delta_Time = 0.00001f;

unsigned int f4size = NBodyCount * 4 * sizeof(float);
unsigned int f3size = NBodyCount * 3 * sizeof(float);


// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

float4* dev_velmass;
float3* dev_nextpos;

float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;
float translate_x = 0.0;
float translate_y = 0.0;

StopWatchInterface *timer = NULL;

int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)
#define dkernel CUDA_KERNEL(ThreadSize, BlockSize)

// declaration, forward
bool runProgram(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);

const char *sSDKsample = "N body cuda simulation";


//Two array hash map
typedef unsigned int gridint;

#define gridSize (gridint(Boundary_Distance) * 2U) //the grid size
#define gridCellSize 1.0f //the size of each cell in the grid

#define gridTotalSize ((gridSize / gridCellSize))

#define gridCellCount (gridint(gridSize * gridSize * gridSize / gridCellSize / gridCellSize / gridCellSize))

#define cellkernel CUDA_KERNEL(16, 500)
#define singlekernel CUDA_KERNEL(1, 1)


// holds the hash value for each particle
gridint* dev_hash_key;
gridint* dev_hash_value;

// holds where in the hash table a cell index starts
gridint* dev_index_start;

// size and position of cells for fast gravitational attraction
gridint* dev_hash_size;
float3* dev_hash_pos3;

#define CellAdder  make_float3(Boundary_Distance, Boundary_Distance, Boundary_Distance)
#define CellMultiplier  make_float3(gridCellSize, gridCellSize, gridCellSize)

__inline__ __device__ gridint getCellIndex(float3 pos)
{
    pos = pos + CellAdder;
    pos = pos / CellMultiplier;

    gridint x = (gridint)(pos.x);
    gridint y = (gridint)(pos.y);
    gridint z = (gridint)(pos.z);

    return x + y * gridSize + z * gridSize * gridSize;
}

__inline__ __device__ float3 getCellPosition(gridint index)
{
	float x = float(index % gridSize);
    float y = float((index / gridSize) % gridSize);
    float z = float(index / gridSize / gridSize);

    float3 pos = make_float3(x, y, z);

    pos = pos * CellMultiplier;
    pos = pos - CellAdder;

    //printf("pos: %f %f %f\n", pos.x, pos.y, pos.z);

	return pos;
}

__global__ void testCellIndexing()
{
    for (int x = -Boundary_Distance+1; x < Boundary_Distance-1 ; x++)
    {
        for (int y = -Boundary_Distance+1; y < Boundary_Distance-1 ; y++)
        {
            for (int z = -Boundary_Distance+1; z < Boundary_Distance-1 ; z++)
            {
                float3 pos = make_float3(x, y, z);
                gridint cellId = getCellIndex(pos);
                float3 pos2 = getCellPosition(cellId);

                float3 diff = pos - pos2;

                if (length(diff) > 0.01f)
                    printf("error: %f %f %f\n", diff.x, diff.y, diff.z);
                    
            }
        }
    }
}

__global__ void hashPopulate_kernel(gridint* hash_key, gridint* hash_value, float4* pos, gridint* hash_size, float3* hash_pos3, gridint* index_start)
{
	gridint particleId = blockIdx.x * blockDim.x + threadIdx.x;

	float3 pos3 = make_float3(pos[particleId].x, pos[particleId].y, pos[particleId].z);
    gridint cellId = getCellIndex(pos3);

	hash_key[particleId] = cellId;
    hash_value[particleId] = particleId;
}

void sortHash()
{
    thrust::sort_by_key(thrust::device, dev_hash_key, dev_hash_key + NBodyCount, dev_hash_value);
}

__global__ void findHashStarts_kernel(gridint* hash_key, gridint* hash_value, gridint* hash_size, float3* hash_pos3, gridint* index_start)
{
	gridint particleId = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned cellId = hash_key[particleId];

	if (particleId == 0)
	{
		index_start[cellId] = 0;
	}
	else if (cellId != hash_key[particleId - 1])
	{
		index_start[cellId] = particleId;

        //update hash size
        gridint count = 1;
        while ( ((particleId + count) <= NBodyCount) && (hash_key[particleId + count] == cellId) )
            count++;

        hash_size[cellId] = count;

        //update hash pos3
        float3 pos3 = getCellPosition(cellId) + make_float3(1, 1, 1) * gridCellSize / 2.0f;

        hash_pos3[cellId] = pos3;

        //if (cellId == 7093)
            //printf("cellpos: %d, %d, %d\n", pos3.x, pos3.y, pos3.z);
	}
}

void createHashBuffers()
{
    checkCudaErrors(cudaMalloc((void**)&dev_hash_key, NBodyCount * sizeof(gridint)));
	checkCudaErrors(cudaMalloc((void**)&dev_hash_value, NBodyCount * sizeof(gridint)));
    checkCudaErrors(cudaMalloc((void**)&dev_index_start, (gridCellCount+1) * sizeof(gridint)));
    checkCudaErrors(cudaMalloc((void**)&dev_hash_size, (gridCellCount+1) * sizeof(gridint)));
    checkCudaErrors(cudaMalloc((void**)&dev_hash_pos3, (gridCellCount+1) * sizeof(float3)));
}

void calculateHash(float4* dptr)
{
    cudaMemset(dev_hash_size, 0U, (gridCellCount+1) * sizeof(gridint));

    hashPopulate_kernel dkernel(dev_hash_key, dev_hash_value, dptr, dev_hash_size, dev_hash_pos3, dev_index_start); CUDASYNC

    sortHash(); CUDASYNC

    findHashStarts_kernel dkernel(dev_hash_key, dev_hash_value, dev_hash_size, dev_hash_pos3, dev_index_start); CUDASYNC
}

__global__ void nbody_kernel(float4* prevpos, float3* nextpos, float4* velocity, float deltatime, float Gravitational_Constant, float Center_Pull, gridint* hash_key, gridint* hash_value, gridint* hash_size, float3* hash_pos3, gridint* index_start)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    float x, y, z, vx, vy, vz, mass;

    // read position and velocity
    x = prevpos[index].x;
    y = prevpos[index].y;
    z = prevpos[index].z;

    vx = velocity[index].x;
    vy = velocity[index].y;
    vz = velocity[index].z;
    mass = velocity[index].w;

    nextpos[index] = make_float3(x, y, z);


#ifdef NAIVE_GRAVITY

    //naive implementation of gravitational attraction,
    for (int i = 0; i < NBodyCount; i++) {
        if (i != index) {
            float dx = prevpos[i].x - x;
            float dy = prevpos[i].y - y;
            float dz = prevpos[i].z - z;
            float distSqr = max(dx * dx + dy * dy + dz * dz, minDist );
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            float s = velocity[i].w * invDist3 * Gravitational_Constant * deltatime;
            vx += dx * s;
            vy += dy * s;
            vz += dz * s;
        }
    }
#else
    //vx += -x * deltatime * Center_Pull;
    //vy += -y * deltatime * Center_Pull;
    //vz += -z * deltatime * Center_Pull;

    //vx *= 0.999f;
    //vy *= 0.999f;
    //vz *= 0.999f;

    x += vx * deltatime;
    y += vy * deltatime;
    z += vz * deltatime;



    // write output vertex
    prevpos[index] = make_float4(x, y, z, 1.0f);
    velocity[index] = make_float4(vx, vy, vz, mass);
}

__global__ void nbody_col_det(float4* prevpos, float3* nextpos, float4* velocity, float deltatime, gridint* hash_key, gridint* hash_value, gridint* hash_size, float3* hash_pos3, gridint* index_start)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    float x, y, z;

    x = prevpos[index].x;
    y = prevpos[index].y;
    z = prevpos[index].z;


#ifdef NAIVE_COLLISIONS

    for (int i = 0; i < NBodyCount; i++) {
        if (i != index) {
            float dx = prevpos[i].x - x;
            float dy = prevpos[i].y - y;
            float dz = prevpos[i].z - z;

            float distSqrt = sqrtf(dx * dx + dy * dy + dz * dz);

            if (distSqrt < minDist) {
                float moveby = (minDist - distSqrt) / distSqrt / 2.0f * Collision_Elasticity;
                x -= dx * moveby;
                y -= dy * moveby;
                z -= dz * moveby;

                prevpos[i].x += dx * moveby;
                prevpos[i].y += dy * moveby;
                prevpos[i].z += dz * moveby;
                //atomicAdd(&prevpos[i], make_float4(dx * moveby, dy * moveby, dz * moveby, 0.0f)); //only in compute 9.x +
            }
        }
    }
    
#else

    //grid based hash map implementation of gravitational attraction
    gridint cellId = getCellIndex(nextpos[index]);
    gridint cellStart = index_start[cellId];
    gridint cellSize = hash_size[cellId];
    gridint cellEnd = cellStart + cellSize;

    for (gridint cellIndex = cellStart; cellIndex < cellEnd; cellIndex++) {
        gridint particleIndex = hash_value[cellIndex];

        if (particleIndex != index) 
        {
            float4 pos = prevpos[particleIndex];
            float dx = pos.x - x;
            float dy = pos.y - y;
            float dz = pos.z - z;
            float distSqrt = sqrtf(dx * dx + dy * dy + dz * dz);

            if (distSqrt < minDist) {
                float moveby = (minDist - distSqrt) / distSqrt / 2.0f * Collision_Elasticity;
                x -= dx * moveby;
                y -= dy * moveby;
                z -= dz * moveby;

                prevpos[particleIndex] = make_float4(pos.x + dx * moveby, pos.y + dy * moveby, pos.z + dz * moveby, 1.0f);
            }
        }
    }

    // check the 4 adjacent cells that's close to the x, y, z
    int fnextX;
    if (fmodf(x + Boundary_Distance, gridCellSize) > gridCellSize / 2)
        fnextX = + 1;
    else 
		fnextX = - 1;

    int fnextY;
    if (fmodf(y + Boundary_Distance, gridCellSize) > gridCellSize / 2)
		fnextY = + gridSize;
	else
		fnextY = - gridSize;

    int fnextZ;
	if (fmodf(z + Boundary_Distance, gridCellSize) > gridCellSize / 2)
        fnextZ = + int(gridSize * gridSize);
    else
        fnextZ = - int(gridSize * gridSize);

    gridint nextXYZ = cellId + fnextX + fnextY + fnextZ;
    gridint nextX = cellId + fnextX;
    gridint nextY = cellId + fnextY;
    gridint nextZ = cellId + fnextZ;

    if (nextXYZ > 0 && nextXYZ < gridSize * gridSize * gridSize) {
        cellStart = index_start[nextXYZ];
        cellSize = hash_size[nextXYZ];
        cellEnd = cellStart + cellSize;

        for (gridint cellIndex = cellStart; cellIndex < cellEnd; cellIndex++) {
            gridint particleIndex = hash_value[cellIndex];

            if (particleIndex != index)
            {
                float4 pos = prevpos[particleIndex];
                float dx = pos.x - x;
                float dy = pos.y - y;
                float dz = pos.z - z;
                float distSqrt = sqrtf(dx * dx + dy * dy + dz * dz);

                if (distSqrt < minDist) {
                    float moveby = (minDist - distSqrt) / distSqrt / 2.0f * Collision_Elasticity;
                    x -= dx * moveby;
                    y -= dy * moveby;
                    z -= dz * moveby;

                    prevpos[particleIndex] = make_float4(pos.x + dx * moveby, pos.y + dy * moveby, pos.z + dz * moveby, 1.0f);
                }
            }
        }
    }

    if (nextX > 0 && nextX < gridSize * gridSize * gridSize) {
        cellStart = index_start[nextX];
        cellSize = hash_size[nextX];
        cellEnd = cellStart + cellSize;

        for (gridint cellIndex = cellStart; cellIndex < cellEnd; cellIndex++) {
            gridint particleIndex = hash_value[cellIndex];

            if (particleIndex != index)
            {
                float4 pos = prevpos[particleIndex];
                float dx = pos.x - x;
                float dy = pos.y - y;
                float dz = pos.z - z;
                float distSqrt = sqrtf(dx * dx + dy * dy + dz * dz);

                if (distSqrt < minDist) {
                    float moveby = (minDist - distSqrt) / distSqrt / 2.0f * Collision_Elasticity;
                    x -= dx * moveby;
                    y -= dy * moveby;
                    z -= dz * moveby;

                    prevpos[particleIndex] = make_float4(pos.x + dx * moveby, pos.y + dy * moveby, pos.z + dz * moveby, 1.0f);
                }
            }
        }
    }

    if (nextY > 0 && nextY < gridSize * gridSize * gridSize) {
        cellStart = index_start[nextY];
        cellSize = hash_size[nextY];
        cellEnd = cellStart + cellSize;

        for (gridint cellIndex = cellStart; cellIndex < cellEnd; cellIndex++) {
            gridint particleIndex = hash_value[cellIndex];

            if (particleIndex != index)
            {
                float4 pos = prevpos[particleIndex];
                float dx = pos.x - x;
                float dy = pos.y - y;
                float dz = pos.z - z;
                float distSqrt = sqrtf(dx * dx + dy * dy + dz * dz);

                if (distSqrt < minDist) {
                    float moveby = (minDist - distSqrt) / distSqrt / 2.0f * Collision_Elasticity;
                    x -= dx * moveby;
                    y -= dy * moveby;
                    z -= dz * moveby;

                    prevpos[particleIndex] = make_float4(pos.x + dx * moveby, pos.y + dy * moveby, pos.z + dz * moveby, 1.0f);
                }
            }
        }
    }

    if (nextZ > 0 && nextZ < gridSize * gridSize * gridSize) {
        cellStart = index_start[nextZ];
        cellSize = hash_size[nextZ];
        cellEnd = cellStart + cellSize;

        for (gridint cellIndex = cellStart; cellIndex < cellEnd; cellIndex++) {
            gridint particleIndex = hash_value[cellIndex];

            if (particleIndex != index)
            {
                float4 pos = prevpos[particleIndex];
                float dx = pos.x - x;
                float dy = pos.y - y;
                float dz = pos.z - z;
                float distSqrt = sqrtf(dx * dx + dy * dy + dz * dz);

                if (distSqrt < minDist) {
                    float moveby = (minDist - distSqrt) / distSqrt / 2.0f * Collision_Elasticity;
                    x -= dx * moveby;
                    y -= dy * moveby;
                    z -= dz * moveby;

                    prevpos[particleIndex] = make_float4(pos.x + dx * moveby, pos.y + dy * moveby, pos.z + dz * moveby, 1.0f);
                }
            }
        }
    }

#endif

    prevpos[index] = make_float4(x, y, z, 1.0f);
}

__global__ void nbody_update_pos(float4* prevpos, float3* nextpos, float4* velocity, float deltatime, float Center_Pull)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    float x, y, z, nx, ny, nz, mass;

    // read position and velocity
    x = prevpos[index].x;
    y = prevpos[index].y;
    z = prevpos[index].z;

    nx = nextpos[index].x;
    ny = nextpos[index].y;
    nz = nextpos[index].z;

    mass = velocity[index].w;

    float vx = (x - nx) / deltatime;
    float vy = (y - ny) / deltatime;
    float vz = (z - nz) / deltatime;

    float dist = sqrtf(x * x + y * y + z * z);

    if (dist > Boundary_Distance) {
        vx = -vx;
        vy = -vy;
        vz = -vz;

        x += 2 * vx * deltatime;
        y += 2 * vy * deltatime;
        z += 2 * vz * deltatime;

        prevpos[index] = make_float4(x, y, z, 1.0f);
    }

    velocity[index] = make_float4(vx, vy, vz, mass);
}

__device__ float randf(float x, float y) {
    float t = sinf(x * 12.9898f + y * 78.233f) * 43758.5453f;
    return (t - floorf(t)) * 2.f - 1.f;
}

__device__ float randf(float x) {
    float t = sinf(x * 12.9898f) * 8.457;
    return t - floorf(t) * 2.0f;
}

__global__ void rand_float4(float4* array, float randseed, float multip)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    float f = float(index);

    float x = randf(f + randseed, 123.57) * multip;
    float y = randf(f + randseed, 23.4) * multip;
    float z = randf(f + randseed, 2.56) * multip;

    while (x * x + y * y + z * z > multip * multip) {
        f += 4.678f;
        x = randf(f + randseed, 123.57) * multip;
        y = randf(f + randseed, 23.4) * multip;
        z = randf(f + randseed, 2.56) * multip;
    }

    y *= 0.5;

    //x = copysignf(x * x, x);
    //y = copysignf(y * y, y) * 0.5;
    //z = copysignf(z * z, z);

    array[index] = make_float4(x, y, z, 1.0f);
}

__global__ void add_rot_speed(float4* posarray, float4* velmass, float Omega_Rot_Speed)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    float x = posarray[index].x;
    //float y = posarray[index].y;
    float z = posarray[index].z;

    float vx = velmass[index].x;
    float vy = velmass[index].y;
    float vz = velmass[index].z;

    float dist = max(sqrtf(x * x + z * z), 0.001f);

    float nx = x / dist;
    float nz = z / dist;

    vx += Omega_Rot_Speed * nz;
    vz -= Omega_Rot_Speed * nx;

    velmass[index] = make_float4(vx, vy, vz, velmass[index].w);
}

float randf() {
    return (float)std::rand() / (float)RAND_MAX;
}

void resetBuffers()
{
    // set random seed
    std::srand(345860);

    float4* velmass = (float4*)malloc(f4size);

    for (int i = 0; i < NBodyCount; i++)
    {
        velmass[i] = make_float4(randf(), randf(), randf(), 1.0f);
    }

    // dev_velmass
    checkCudaErrors(cudaMalloc((void**)&dev_velmass, f4size));
    checkCudaErrors(cudaMemcpy(dev_velmass, velmass, f4size, cudaMemcpyHostToDevice));

    // dev_nextpos
    checkCudaErrors(cudaMalloc((void**)&dev_nextpos, f4size));

    createHashBuffers();

    // map OpenGL buffer object for writing from CUDA
    float4* dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_vbo_resource));

    rand_float4 dkernel(dptr, 9.5f, 6.0f);

    rand_float4 dkernel(dev_velmass, 2.1f, 0.1f);

    add_rot_speed dkernel(dptr, dev_velmass, Omega_Rot_Speed);

    calculateHash(dptr);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

    // run the cuda part
    runCuda(&cuda_vbo_resource);

}

bool runProgram(int argc, char** argv, char* ref_file)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char**)argv);


    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return false;
    }

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif


    //checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (NBodyCount * 2) * sizeof(float3)));

    // create VBO
    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

    resetBuffers();

    testCellIndexing singlekernel(); CUDASYNC

    // start rendering mainloop
    glutMainLoop();

    return true;
}

int main(int argc, char** argv)
{
    char* ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char**)argv, "file"))
        {
            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
            getCmdLineArgumentString(argc, (const char**)argv, "file", (char**)&ref_file);
        }
    }

    printf("\n");

    runProgram(argc, argv, ref_file);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "FPS: %3.5f (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

void reshapeWindow(int w, int h)
{
    window_width = (unsigned int)(w);
    window_height = (unsigned int)(h);
    glViewport(0, 0, window_width, window_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1f, 1000.0);

    SDK_CHECK_ERROR_GL();
}

bool initGL(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // initialize necessary OpenGL extensions
    if (!isGLVersionSupported(2, 0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glutReshapeFunc(reshapeWindow);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1f, 1000.0);

    SDK_CHECK_ERROR_GL();

    return true;
}

void runCuda(struct cudaGraphicsResource** vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4* dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource));


    for (int i = 0; i < 1; i++) {
        calculateHash(dptr);

        nbody_kernel dkernel(dptr, dev_nextpos, dev_velmass, Delta_Time, Gravitational_Constant, Center_Pull, dev_hash_key, dev_hash_value, dev_hash_size, dev_hash_pos3, dev_index_start); CUDASYNC
        

#ifdef DO_COLLISIONS
            for (int j = 0; j < Collision_Iterations; j++) {
                nbody_col_det dkernel(dptr, dev_nextpos, dev_velmass, Delta_Time, dev_hash_key, dev_hash_value, dev_hash_size, dev_hash_pos3, dev_index_start); CUDASYNC
            }
#endif 

        nbody_update_pos dkernel(dptr, dev_nextpos, dev_velmass, Delta_Time, Center_Pull); CUDASYNC
    }

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    glBufferData(GL_ARRAY_BUFFER, f4size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW); 

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, NBodyCount);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case 'r':
            resetBuffers(); 
            printf("Resetting buffers\n");
            break;
        case 'w':
            Delta_Time *= 1.25f;
            break;
        case 's':
            Delta_Time /= 1.25f;
            break;
        case (27):
#if defined(__APPLE__) || defined(MACOSX)
            exit(EXIT_SUCCESS);
            break;
#else
            glutDestroyWindow(glutGetWindow());
            break;
#endif
    }
}

float3 matMul(GLfloat* mat, float3 vec)
{
	float x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
    float y = mat[4] * vec.x + mat[5] * vec.y + mat[6] * vec.z;
    float z = mat[8] * vec.x + mat[9] * vec.y + mat[10] * vec.z;

    return make_float3(x, y, z);
}

void glMatVecMul(float3 vec)
{
    GLfloat mat[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, mat);

    vec = matMul(mat, vec);
    glTranslatef(vec.x, vec.y, vec.z);

}

void glRotateF3(float3 vec)
{
    GLfloat mat[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, mat);

    GLfloat pos[3] = { mat[3], mat[7], mat[11] };
    glTranslatef(-pos[0], -pos[1], -pos[2]);

    glRotatef(vec.x, mat[0], mat[4], mat[8]);
    glRotatef(vec.y, mat[1], mat[5], mat[9]);
    glRotatef(vec.z, mat[2], mat[6], mat[10]);

    glTranslatef(pos[0], pos[1], pos[2]);
}

void mouse(int button, int state, int x, int y)
{
    if ((button == 3) || (button == 4)) // It's a wheel event
    {
        // Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
        if (state == GLUT_UP) return; // Disregard redundant GLUT_UP events

        bool isup = (button == 3);

        const float amount = 0.2f;

        if (isup)
        {
			translate_z *= amount;

            glMatVecMul(make_float3(0, 0, amount));
		}
        else
        {
			translate_z /= amount;

            glMatVecMul(make_float3(0, 0, -amount));
		}
    }

    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
        //glRotatef(dx * 0.2f, 0.0, 1.0, 0.0);
        //glRotatef(dy * 0.2f, 1.0, 0.0, 0.0);
        glRotateF3(make_float3(dy * 0.2f, dx * 0.2f, 0));
    }
    else if (mouse_buttons & 4)
    {
        translate_x += dx * 0.01f;
        translate_y += -dy * 0.01f;
        //glTranslatef(dx * 0.01f, -dy * 0.01f, 0.0f);
        glMatVecMul(make_float3(dx * 0.01f, -dy * 0.01f, 0.0f));
    }

    mouse_old_x = x;
    mouse_old_y = y;
}
