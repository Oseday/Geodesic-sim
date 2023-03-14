#include <stdio.h>
#include <cmath>

struct float3 {
	float x, y, z;
};

static float3 make_float3(float x, float y, float z) noexcept {
    return float3{x,y,z};
}

static float3 operator+(float3 a, float3 b) noexcept
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static float3 operator-(float3 a, float3 b) noexcept
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static float3 operator*(float3 a, float3 b) noexcept
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static float3 operator/(float3 a, float3 b) noexcept
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

typedef unsigned int gridint;

#define toint(a) (static_cast<gridint>(a))
#define tofloat(a) (static_cast<float>(a))

#define Boundary_Distance 10.0f
#define gridSize (gridint(Boundary_Distance) * 4U) //the grid size
#define gridCellSize 0.5f //the size of each cell in the grid

#define gridTotalSize 0 
//((gridSize / gridCellSize))

const float3 addr = make_float3(Boundary_Distance, Boundary_Distance, Boundary_Distance);
const float3 mult = make_float3(gridCellSize, gridCellSize, gridCellSize);

gridint getCellIndex(float3 pos)
{
    //printf("posb: %f %f %f\n", pos.x, pos.y, pos.z);

    pos = pos + addr;
    pos = pos / mult;

    //printf("posa: %f %f %f\n" , pos.x, pos.y, pos.z);

    gridint x = toint( pos.x );
    gridint y = toint( pos.y );
    gridint z = toint( pos.z );

    return x + y * gridSize + z * gridSize * gridSize;
}

float3 getCellPosition(gridint index)
{
    const float x = tofloat(index % gridSize);
    const float y = tofloat((index / gridSize) % gridSize);
    const float z = tofloat(index / gridSize / gridSize);

    float3 pos = make_float3(x, y, z);

    //printf("posb: %f %f %f\n", pos.x, pos.y, pos.z);

    pos = pos * mult;
    pos = pos - addr;


    return pos;
}

void _testCellIndexing()
{
    printf("gridSize: %u\n", gridSize);
    printf("gridCellSize: %f\n", gridCellSize);
    printf("gridTotalSize: %f\n", gridTotalSize);

    for (int x = -Boundary_Distance + 1; x < Boundary_Distance-1; x++)
    {
        for (int y = -Boundary_Distance + 1; y < Boundary_Distance - 1; y++)
        {
            for (int z = -Boundary_Distance + 1; z < Boundary_Distance - 1; z++)
            {
                float3 pos = make_float3(x, y, z);

                gridint cellId = getCellIndex(pos);
                float3 pos2 = getCellPosition(cellId);


                const float3 diff = pos - pos2;

                const float length = std::abs(diff.x) + abs(diff.y) + abs(diff.z);

                //printf("pos1: %2.0f %2.0f %2.0f\n", pos.x, pos.y, pos.z);
                //printf("pos2: %2.0f %2.0f %2.0f\n", pos2.x, pos2.y, pos2.z);

                if (length > 0.0001f)
                    printf("error: %f %f %f\n", diff.x, diff.y, diff.z);


            }
        }
    }

    
}

int _main()
{
	_testCellIndexing();
}