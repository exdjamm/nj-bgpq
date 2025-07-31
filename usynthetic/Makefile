nvcc = nvcc
nvccflag = -std=c++11 -O3 -arch=native
nvcc_infoflag = -std=c++11 -g -G -arch=native
heappath = ../new_impl

all: uheap_synthetic uheap_synthetic_debug

uheap_synthetic: sort.cu $(heappath)/heap.cuh
	$(nvcc) $(nvccflag) $(testflag) -I$(heappath)/ $< -o $@.run

uheap_synthetic_debug: sort.cu $(heappath)/heap.cuh
	$(nvcc) $(nvcc_infoflag) $(testflag) -I$(heappath)/ $< -o $@.run

clean:
	rm -rf uheap_synthetic.run uheap_synthetic_debug.run
