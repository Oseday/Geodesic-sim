/*

#define gridSize (unsigned int(Boundary_Distance * 2)) //the grid size
#define gridCellSize 1.0f //the size of each cell in the grid
#define cellSizeMultiplier 8 //the multiplier for the cell size, used to resize the cell if it is full
#define cellSizeStart 128 //the multiplier for the cell size, used to resize the cell if it is full

#define gridCellCount (unsigned int(gridSize * gridSize * gridSize / gridCellSize / gridCellSize / gridCellSize))

#define cellkernel CUDA_KERNEL(16, 500)
#define singlekernel CUDA_KERNEL(1, 1)

struct Cell {
    float3 Position;
    unsigned int Count;
    unsigned int CountTotal;
    float3* cellStart;
    float3* cellEnd;
};

__inline__ __device__ unsigned int getCellIndex(float3 pos)
{
    unsigned int x = (unsigned int)(pos.x / gridCellSize + gridSize / 2);
    unsigned int y = (unsigned int)(pos.y / gridCellSize + gridSize / 2);
    unsigned int z = (unsigned int)(pos.z / gridCellSize + gridSize / 2);

    return x + y * gridSize + z * gridSize * gridSize;
}

__inline__ __device__ float3 getCellPosition(unsigned int index)
{
    unsigned int x = index % gridSize;
    unsigned int y = (index / gridSize) % gridSize;
    unsigned int z = index / gridSize / gridSize;

    return make_float3((x - gridSize / 2) * gridCellSize, (y - gridSize / 2) * gridCellSize, (z - gridSize / 2) * gridCellSize);
}

//unsigned int atomicAdd(unsigned int* address, unsigned int val);

// adds a point to the cell
// if the cell is full, it will be resized
__inline__ __device__ void addPointCell(Cell* cells, float3 pos)
{


    unsigned int cellIndex = getCellIndex(pos);
    Cell* cell = cells + cellIndex;

    //unsigned int currentCount = atomicAdd(&(cell->Count), 1);
    unsigned int currentCount = cell->Count;

    currentCount++;

    cell->Count = currentCount;


    if (currentCount >= cell->CountTotal)
    {
        // resize the cell
        unsigned int newCount = cell->CountTotal * cellSizeMultiplier;
        float3* newStart = (float3*)malloc(newCount * sizeof(float3));
        float3* newEnd = newStart + newCount;

        memcpy(newStart, cell->cellStart, cell->CountTotal * sizeof(float3));
        free(cell->cellStart);

        cell->cellStart = newStart;
        cell->cellEnd = newEnd;
        cell->CountTotal = newCount;
    }


    //cell->cellStart[currentCount-1] = pos;
}

__inline__ __device__ void clearCell(Cell* cell)
{
    cell->Count = 0;
    cell->CountTotal = 0;
    cell->cellStart = (float3*)malloc(cellSizeStart * sizeof(float3));
    cell->cellEnd = cell->cellStart;
}

__global__ void clearCells(Cell* cells)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    clearCell(cells + index);
}

__global__ void calculatePositionsCellsArray(Cell* cells)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    cells[index].Position = getCellPosition(index);
}

__global__ void addPointsToCellsParallel(Cell* cells, float4* positions)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    float4 pos = positions[index];

    addPointCell(cells, make_float3(pos.x, pos.y, pos.z));
}

__global__ void addPointsToCellsSerial(Cell* cells, float4* positions)
{

    for (unsigned int i = 0; i < NBodyCount; i++)
    {
        float4 pos = positions[i];
        addPointCell(cells, make_float3(pos.x, pos.y, pos.z));
    }
}

Cell* dev_cells;

*/