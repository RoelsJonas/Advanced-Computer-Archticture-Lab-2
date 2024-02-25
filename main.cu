#include <iostream>
#include <chrono>


__global__ void getMaxUsingAtomicFunction(int* in) {
    atomicMax(&in[0], in[blockIdx.x * blockDim.x + threadIdx.x]);
}


__global__ void getMaxUsingReduction(int* in, int size) {
    int step = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    while(i % step == 0 && step < size) {
//        printf("Comparing: %d and %d, step: %d on i: %d\n", 2*i, 2*i + step, step, i);
        atomicMax(&in[2*i], in[2*i + step]);
        step *= 2;
        __syncthreads();
    }
}

__global__ void getMaxUsingReductionFast(int* in, int size) {
    int step = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    while(i % step == 0 && step < size) {
//        printf("Comparing: %d and %d, step: %d on i: %d\n", 2*i, 2*i + step, step, i);
//        atomicMax(&in[2*i], in[2*i + step]);
        if(in[2*i] < in[2*i + step]) in[2*i] = in[2*i + step];
        step *= 2;
        __syncthreads();
    }
}

void findMaxUsingCPU(int size) {
    int* temp = (int*)(malloc(size * sizeof(int)));
    for(int i = 0; i < size; i++) {
        temp[i] = i;
    }

    const auto startc = std :: chrono :: steady_clock :: now () ;

    int biggest = -INFINITY;
    for(int i = 0; i < size; i++) {
        if(biggest < temp[i]){
            biggest = temp[i];
        }
    }

    const auto end = std :: chrono :: steady_clock :: now () ;
    const std :: chrono :: duration<double> elapsed_seconds{end - startc};
    std::cout << size << ";" << elapsed_seconds.count()*1000 << ";";

    free(temp);
}

void findMaxUsingReduction(int size) {
    int* temp = (int*)(malloc(size * sizeof(int)));
    for(int i = 0; i < size; i++) {
        temp[i] = i;
    }

    int* input;
    cudaMalloc(&input, size*sizeof(int));
    cudaMemcpy( input, temp, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start);


    getMaxUsingReduction<<<1, size / 2>>>(input, size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << size << ";" << milliseconds << std::endl;
//    cudaMemcpy( temp, input, size * sizeof(int), cudaMemcpyDeviceToHost);
//    std::cout << "Results: " << std::endl;
//    for(int i = 0; i < size; i++) std::cout << temp[i] << ", ";
//    std::cout << std::endl;

    free(temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
}

void findMaxUsingReductionFast(int size) {
    int* temp = (int*)(malloc(size * sizeof(int)));
    for(int i = 0; i < size; i++) {
        temp[i] = i;
    }

    int* input;
    cudaMalloc(&input, size*sizeof(int));
    cudaMemcpy( input, temp, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start);


    getMaxUsingReductionFast<<<1, size / 2>>>(input, size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << size << ";" << milliseconds << std::endl;
//    cudaMemcpy( temp, input, size * sizeof(int), cudaMemcpyDeviceToHost);
//    std::cout << "Results: " << std::endl;
//    for(int i = 0; i < size; i++) std::cout << temp[i] << ", ";
//    std::cout << std::endl;

    free(temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
}

void findMaxUsingAtomics(int size) {
    int* temp = (int*)(malloc(size * sizeof(int)));
    for(int i = 0; i < size; i++) {
        temp[i] = i;
    }

    int* input;
    cudaMalloc(&input, size*sizeof(int));
    cudaMemcpy( input, temp, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start);
    const auto startc = std :: chrono :: steady_clock :: now () ;

    getMaxUsingAtomicFunction<<<1, size>>>(input);

    const auto end = std :: chrono :: steady_clock :: now () ;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    const std :: chrono :: duration<double> elapsed_seconds{end - startc};

    std::cout << size << ";" << milliseconds << ";";
//    cudaMemcpy( temp, input, size * sizeof(int), cudaMemcpyDeviceToHost);
//    std::cout << "Results: " << std::endl;
//    for(int i = 0; i < size; i++) std::cout << temp[i] << ", ";
//    std::cout << std::endl;

    free(temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
}

int main() {
//    int size = 2000;
    findMaxUsingCPU(1);
    findMaxUsingAtomics(1);
    findMaxUsingReduction(1);
    findMaxUsingReductionFast(1);

    for(int size = 1; size < 2048; size++) {
        findMaxUsingCPU(size);
        findMaxUsingAtomics(size);
        findMaxUsingReduction(size);
        findMaxUsingReductionFast(size);
    }
    return 0;
}



