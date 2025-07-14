nvcc = nvcc
nvccflag = -std=c++11 -O3 -arch=sm_61 -DHEAP_SORT
heappath = ../new_impl

all: uheap_synthetic

uheap_synthetic: sort.cu $(heappath)/heap.cuh
	$(nvcc) $(nvccflag) $(testflag) -I$(heappath)/ $< -o $@.run

clean:
	rm -rf out
