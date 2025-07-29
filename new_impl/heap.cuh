#ifndef _H_HEAP_
#define _H_HEAP_

#define AVAIL 0
#define INSHOLD 1
#define DELMOD 2
#define INUSE 3

#include "./uheaputil.cuh"

using namespace std;

__device__ __inline__ int atomicForceInuse(int *status, int new_value);

template <typename K, typename A>
class KAuxHeap
{
public:
    K k_init_limit;
    A aux_init_limit;

    int max_batch_num;
    int batch_size;

    int *batch_count;

    K *heap;
    A *heap_aux;
    int *batch_status;

    KAuxHeap(int batch_num, int batch_s, K k_limit, A aux_limit) : max_batch_num(batch_num), batch_size(batch_s), k_init_limit(k_limit), aux_init_limit(aux_limit)
    {
        cudaMalloc(&batch_count, sizeof(int));
        // cudaMalloc(&partial_buffer_size, sizeof(int));

        cudaMalloc(&heap, sizeof(K) * (max_batch_num)*batch_size);
        cudaMalloc(&heap_aux, sizeof(A) * (max_batch_num)*batch_size);
        cudaMalloc(&batch_status, sizeof(int) * (max_batch_num));

        cudaMemset(batch_count, 0, sizeof(int));
        // cudaMemset(partial_buffer_size, 0, sizeof(int));

        cudaMemset(batch_status, AVAIL, sizeof(int) * (max_batch_num));

        // Heap and Aux is set by outside kernel defined at of file's header;
    }

    ~KAuxHeap()
    {
        // cudaFree(partial_buffer_size);
        cudaFree(batch_count);

        cudaFree(heap);
        cudaFree(heap_aux);
        cudaFree(batch_status);

        max_batch_num = 0;
        batch_size = 0;
    }

    __device__ bool atomicChangeStatus(int *status, int oriS, int newS)
    {
        if ((oriS == AVAIL && newS == INUSE) ||
            (oriS == INUSE && newS == AVAIL) ||
            (oriS == INUSE && newS == INSHOLD) ||
            (oriS == INSHOLD && newS == INUSE) ||
            (oriS == INSHOLD && newS == DELMOD) ||
            (oriS == DELMOD && newS == INUSE) ||
            (oriS == INUSE && newS == DELMOD))
        {
            while (atomicCAS(status, oriS, newS) != oriS)
            {
            }
            return true;
        }
        else
        {
            printf("LOCK ERROR ori %d new %d\n", oriS, newS);
            return false;
        }
    }

    __device__ int has_between(K *first, K *second)
    {
        K minp, maxp, minq, maxq;
        int lesser, intersection;
        lesser = intersection = 0;

        minp = first[0];
        maxp = first[batch_size - 1];

        minq = second[0];
        maxq = second[batch_size - 1];
        __syncthreads();

        if (minq < minq)
            lesser = 1;

        if (minq < maxp || maxq < minp || maxq < maxp)
            intersection = 1;

        return lesser || intersection;
    }

    __device__ bool delete_root(K *items, A *aux_items, int &size)
    {
        // Check Emptyness
        if (*batch_count == 0)
        {
            size = 0;
            return false;
        }

        if (threadIdx.x == 0)
        {
            atomicChangeStatus(&batch_status[0], AVAIL, INUSE);
        }
        __syncthreads();

        size = batch_size;

        batchCopy(items, heap, aux_items, heap_aux, batch_size);

        return true;
    }

    // top -> bottom
    __device__ void delete_update(int smOffset)
    {
        extern __shared__ int s[];
        int *tmpOldIndex = &s[smOffset];

        K *items_sm = (K *)&s[smOffset];
        A *items_aux_sm = (A *)&items_sm[3 * batch_size];
        smOffset += (sizeof(A) + sizeof(K)) * 3 * batch_size / sizeof(int);

        K *items_1 = items_sm;
        K *items_2 = items_sm + batch_size;
        K *items_3 = items_sm + 2 * batch_size;

        K *items_aux_1 = items_aux_sm;
        K *items_aux_2 = items_aux_sm + batch_size;
        K *items_aux_3 = items_aux_sm + 2 * batch_size;

        int lastIndex;

        if (threadIdx.x == 0)
        {
            *tmpOldIndex = atomicSub(batch_count, 1) - 1;
        }
        __syncthreads();

        // There is no more batchs
        if (*tmpOldIndex <= 1)
        {
            if (threadIdx.x == 0)
            {
                atomicChangeStatus(&batch_status[0], INUSE, AVAIL);
            }
            __syncthreads();
            return;
        }

        lastIndex = *tmpOldIndex;
        __syncthreads();

        // Force status INUSE on last batch
        if (threadIdx.x == 0)
        {
            atomicForceInuse(&batch_status[lastIndex], INUSE);
        }
        __syncthreads();

        // Copy lastIndex to SM
        batchCopy(items_1, heap + lastIndex * batch_size,
                  items_aux_1, heap_aux + lastIndex * batch_size,
                  batch_size, true, k_init_limit);

        if (threadIdx.x == 0)
        {
            atomicChangeStatus(&batch_status[lastIndex], INUSE, AVAIL);
        }
        __syncthreads();

        int currentIdx = 0;
        int curPrevStatus = AVAIL;

        int left_idx = 2 * currentIdx + 1;
        int right_idx = 2 * currentIdx + 2;
        int less;

        int prev_status_left, prev_status_right;
        prev_status_left = prev_status_right = INUSE;

        if (left_idx >= *batch_count)
        {
            __syncthreads();
            batchCopy(heap + currentIdx * batch_size, items_1,
                      heap_aux + currentIdx * batch_size, items_aux_1,
                      batch_size);

            if (threadIdx.x == 0)
            {
                atomicChangeStatus(&batch_status[currentIdx], INUSE, AVAIL);
            }
            __syncthreads();
            return;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            prev_status_left = atomicForceInuse(&batch_status[left_idx], INUSE);

            if (prev_status_left == INSHOLD)
                prev_status_left = DELMOD;
        }
        __syncthreads();

        batchCopy(items_2, heap + left_idx * batch_size,
                  items_aux_2, heap_aux + left_idx * batch_size,
                  batch_size);

        if (right_idx >= *batch_count)
        {
            __syncthreads();
            imergePath(items_1, items_2, heap + currentIdx * batch_size, heap + batch_size * left_idx,
                       items_aux_1, items_aux_2, heap_aux + currentIdx * batch_size, heap_aux + left_idx * batch_size,
                       batch_size, smOffset);

            __syncthreads();

            if (threadIdx.x == 0)
            {
                // atomicChangeStatus(&batch_status[left_idx], INUSE, AVAIL);
                atomicChangeStatus(&batch_status[currentIdx], INUSE, curPrevStatus);
                atomicChangeStatus(&batch_status[left_idx], INUSE, prev_status_left);
            }
            __syncthreads();

            return;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            prev_status_right = atomicForceInuse(&batch_status[right_idx], INUSE);

            if (prev_status_right == INSHOLD)
                prev_status_right = DELMOD;
        }
        __syncthreads();

        batchCopy(items_3, heap + right_idx * batch_size,
                  items_aux_3, heap_aux + right_idx * batch_size,
                  batch_size);

        less = has_between(items_1, items_2) || has_between(items_1, items_3);
        __syncthreads();

        int target, prev_status_target;   // Stay with greaters
        int new_idx, prev_status_new_idx; // Goes down
        while (less)
        {
            target = items_2[batch_size - 1] < items_3[batch_size - 1] ? right_idx : left_idx;
            prev_status_target = (target == left_idx) ? prev_status_left : prev_status_right;

            new_idx = (target == left_idx) ? right_idx : left_idx;
            prev_status_new_idx = (new_idx == left_idx) ? prev_status_right : prev_status_left;

            __syncthreads();

            // Order between left and right, greaters go to target
            imergePath(items_2, items_3, items_2, heap + batch_size * target,
                       items_aux_2, items_aux_3, items_aux_2, heap_aux + target * batch_size,
                       batch_size, smOffset);

            if (threadIdx.x == 0)
            {
                atomicChangeStatus(&batch_status[target], INUSE, prev_status_target);
            }
            __syncthreads();

            /* CASOS ENTRE ITEM_1 E 2 */
            if (items_1[0] >= items_2[batch_size - 1])
            {
                __syncthreads();
                batchCopy(heap + currentIdx * batch_size, items_2,
                          heap_aux + currentIdx * batch_size, items_aux_2, batch_size);
            }
            else if (items_1[batch_size - 1] < items_2[0])
            {
                __syncthreads();
                batchCopy(heap + currentIdx * batch_size, items_1,
                          heap_aux + currentIdx * batch_size, items_aux_1, batch_size);
                batchCopy(heap + new_idx * batch_size, items_2,
                          heap_aux + new_idx * batch_size, items_aux_2, batch_size);
                if (threadIdx.x == 0)
                {
                    atomicChangeStatus(&batch_status[currentIdx], INUSE, curPrevStatus);
                    atomicChangeStatus(&batch_status[new_idx], INUSE, prev_status_new_idx);
                }
                __syncthreads();
                break;
            }
            else
            {
                __syncthreads();
                imergePath(items_1, items_2, heap + currentIdx * batch_size, items_1,
                           items_aux_1, items_aux_2, heap_aux + currentIdx * batch_size, items_aux_1,
                           batch_size, smOffset);
            }

            if (threadIdx.x == 0)
            {
                atomicChangeStatus(&batch_status[currentIdx], INUSE, curPrevStatus);
            }
            __syncthreads();

            currentIdx = new_idx;
            curPrevStatus = prev_status_new_idx;

            left_idx = currentIdx * 2 + 1;
            right_idx = currentIdx * 2 + 2;

            prev_status_left = prev_status_right = INUSE;

            if (left_idx >= *batch_count)
            {
                __syncthreads();

                batchCopy(heap + currentIdx * batch_size, items_1,
                          heap_aux + currentIdx * batch_size, items_aux_1,
                          batch_size);

                if (threadIdx.x == 0)
                {
                    atomicChangeStatus(&batch_status[currentIdx], INUSE, curPrevStatus);
                }
                __syncthreads();
                break;
            }
            __syncthreads();

            if (threadIdx.x == 0)
            {
                prev_status_left = atomicForceInuse(&batch_status[left_idx], INUSE);

                if (prev_status_left == INSHOLD)
                    prev_status_left = DELMOD;
            }
            __syncthreads();

            batchCopy(items_2, heap + left_idx * batch_size,
                      items_aux_2, heap_aux + left_idx * batch_size,
                      batch_size);

            if (right_idx >= *batch_count)
            {
                __syncthreads();
                imergePath(items_1, items_2, heap + currentIdx * batch_size, heap + batch_size * left_idx,
                           items_aux_1, items_aux_2, heap_aux + currentIdx * batch_size, heap_aux + left_idx * batch_size,
                           batch_size, smOffset);

                __syncthreads();

                if (threadIdx.x == 0)
                {
                    atomicChangeStatus(&batch_status[left_idx], INUSE, prev_status_left);
                    atomicChangeStatus(&batch_status[currentIdx], INUSE, curPrevStatus);
                }
                __syncthreads();

                break;
            }

            if (threadIdx.x == 0)
            {
                prev_status_right = atomicForceInuse(&batch_status[right_idx], INUSE);

                if (prev_status_right == INSHOLD)
                    prev_status_right = DELMOD;
            }
            __syncthreads();

            batchCopy(items_3, heap + right_idx * batch_size,
                      items_aux_3, heap_aux + right_idx * batch_size,
                      batch_size);

            less = has_between(items_1, items_2) || has_between(items_1, items_3);
            __syncthreads();
        }
    }

    __device__ void insertion(K *items, A *aux, int size, int smOffset)
    {
        extern __shared__ int s[];
        int *tmpOldIndex = &s[smOffset];

        K *items_sm = (K *)&s[smOffset];
        A *items_aux_sm = (A *)&items_sm[2 * batch_size];
        smOffset += (sizeof(A) + sizeof(K)) * 2 * batch_size / sizeof(int);

        K *items_1 = items_sm;
        K *items_2 = items_sm + batch_size;

        K *items_aux_1 = items_aux_sm;
        K *items_aux_2 = items_aux_sm + batch_size;

        for (int i = threadIdx.x; i < batch_size; i += blockDim.x)
        {
            if (i < size)
            {
                items_1[i] = items[i];
                items_aux_1[i] = aux[i];
            }
            else
            {
                items_1[i] = k_init_limit;
                items_aux_1[i] = aux_init_limit;
            }
        }
        __syncthreads();

        ibitonicSort(items_1, items_aux_1, batch_size);
        __syncthreads();

        if (threadIdx.x == 0)
        {
            *tmpOldIndex = atomicAdd(batch_count, 1);
            atomicChangeStatus(&batch_status[*tmpOldIndex], AVAIL, INUSE);
        }
        __syncthreads();

        int current_idx = *tmpOldIndex;
        int father_idx = current_idx / 2;
        __syncthreads();

        batchCopy(heap + current_idx * batch_size, items_1,
                  heap_aux + current_idx * batch_size, items_aux_1,
                  batch_size);

        if (current_idx == 0)
        {

            if (threadIdx.x == 0)
            {
                atomicChangeStatus(&batch_status[current_idx], INUSE, AVAIL);
            }

            __syncthreads();
            return;
        }

        if (threadIdx.x == 0)
        {
            atomicChangeStatus(&batch_status[current_idx], INUSE, INSHOLD);
        }
        __syncthreads();

        int lesser = has_between(heap + father_idx * batch_size, heap + current_idx * batch_size);
        int run = lesser && (current_idx != father_idx);

        while (run)
        {
            if (threadIdx.x == 0)
            {
                atomicChangeStatus(&batch_status[father_idx], AVAIL, INUSE);
                atomicForceInuse(&batch_status[current_idx], INUSE);
            }
            __syncthreads();

            batchCopy(items_1, heap + current_idx * batch_size,
                      items_aux_1, heap_aux + current_idx * batch_size,
                      batch_size);
            batchCopy(items_2, heap + father_idx * batch_size,
                      items_aux_2, heap_aux + father_idx * batch_size,
                      batch_size);

            imergePath<K, A>(items_1, items_2, heap + father_idx * batch_size, heap + current_idx * batch_size,
                             items_aux_1, items_aux_2, heap_aux + father_idx * batch_size, heap_aux + current_idx * batch_size,
                             batch_size, smOffset);

            __syncthreads();

            if (threadIdx.x == 0)
            {
                atomicChangeStatus(&batch_status[father_idx], INUSE, INSHOLD);
                atomicChangeStatus(&batch_status[current_idx], INUSE, AVAIL);
            }
            __syncthreads();

            current_idx = father_idx;
            father_idx = current_idx / 2;
            lesser = has_between(heap + father_idx * batch_size, heap + current_idx * batch_size);
            run = lesser && (current_idx != father_idx);
            __syncthreads();
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            atomicChangeStatus(&batch_status[current_idx], INSHOLD, INUSE);
            atomicChangeStatus(&batch_status[current_idx], INUSE, AVAIL);
        }
        __syncthreads();
    }
};

__device__ __inline__ int atomicForceInuse(int *status, int new_value)
{
    int value = INUSE;

    while (value == INUSE)
    {
        value = atomicMax(status, INUSE);
    }
    return value;
}

template <typename K, typename A>
__global__ void resetKAuxHeap(KAuxHeap<K, A> heap);

#endif