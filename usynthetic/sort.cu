#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <time.h>
#include <queue>
#include <algorithm>
#include <functional>

#include <heap.cuh>
#include <cmath>
#include "util.hpp"

using namespace std;

__global__ void insertKernel(KAuxHeap<int, int> *heap,
                             int *items,
                             int *aux_items,
                             int arraySize,
                             int batchSize)
{

    //    batchSize /= 3;
    int batchNeed = arraySize / batchSize;
    // insertion
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x)
    {
        // insert items to buffer
        heap->insertion(items + i * batchSize,
                        aux_items + i * batchSize,
                        batchSize, 0);
        __syncthreads();
    }
}

__global__ void deleteKernel(KAuxHeap<int, int> *heap, int *items, int *aux_items, int arraySize, int batchSize)
{

    int batchNeed = arraySize / batchSize;
    int size = 0;
    // deletion
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x)
    {

        // delete items from heap
        if (heap->delete_root(items, aux_items, size) == true)
        {
            __syncthreads();

            heap->delete_update(0);
        }
        __syncthreads();
    }
}

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

int main(int argc, char *argv[])
{

    if (argc != 4)
    {
        cout << argv[0] << " [test type: 0] [# keys in M] [keyType: 0:random 1:ascend 2:descend]\n";
        return -1;
    }

    srand(time(NULL));

    int testType = atoi(argv[1]);
    int arrayNum = atoi(argv[2]);
    int keyType = atoi(argv[3]);

    struct timeval startTime;
    struct timeval endTime;
    double insertTime, deleteTime;
    int batchNum;

    int blockNum = 32;
    int blockSize = 512;
    int batchSize = 1024;

    arrayNum = ((arrayNum + batchSize - 1) / batchSize) * batchSize;
    batchNum = arrayNum / batchSize; // talvez

    int *oriItems = new int[arrayNum];

    srand(0);

    for (int i = 0; i < arrayNum; ++i)
    {
        oriItems[i] = rand() % INT_MAX;
    }

    if (keyType == 1)
    {
        std::sort(oriItems, oriItems + arrayNum);
    }
    else if (keyType == 2)
    {
        std::sort(oriItems, oriItems + arrayNum, std::greater<int>());
    }

    // bitonic heap sort
    KAuxHeap<int, int> h_heap(batchNum, batchSize, INT_MAX, -1);

    int *heapItems;
    int *auxItems;
    KAuxHeap<int, int> *d_heap;

    cudaMalloc((void **)&heapItems, sizeof(int) * (arrayNum));
    cudaMemcpy(heapItems, oriItems, sizeof(int) * (arrayNum), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&auxItems, sizeof(int) * (arrayNum));
    cudaMemcpy(auxItems, oriItems, sizeof(int) * (arrayNum), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_heap, sizeof(KAuxHeap<int, int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(KAuxHeap<int, int>), cudaMemcpyHostToDevice);

    int smemSize = batchSize * 3 * sizeof(int) + batchSize * 3 * sizeof(int);
    smemSize += (blockSize + 1) * sizeof(int) + 2 * batchSize * sizeof(int) + 2 * batchSize * sizeof(int);

    printf("[START] insertion.\n");

    // concurrent insetion
    setTime(&startTime);

    insertKernel<<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, auxItems, arrayNum, batchSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    setTime(&endTime);
    insertTime = getTime(&startTime, &endTime);

    printf("[END] insertion.\n");

    // concurrent deletion
    printf("[START] deletion.\n");
    setTime(&startTime);

    deleteKernel<<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, auxItems, arrayNum, batchSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    setTime(&endTime);
    printf("[END] deletion.\n");
    deleteTime = getTime(&startTime, &endTime);

    printf("%s,insdel,%d,%d,%.2f,%.2f,%.2f\n",
           argv[0],
           keyType, arrayNum, insertTime, deleteTime, insertTime + deleteTime);

    cudaFree(heapItems);
    cudaFree(auxItems);
    heapItems = NULL;
    cudaFree(d_heap);
    d_heap = NULL;

    return 0;
}
