#include <nccl.h> 
#include <cuda_runtime.h> 
#include <iostream>
#include <vector>
#include <algorithm> // for sorting
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>

/*Thrust is a parallel algorithms library designed to work with GPUs.
Thrust provides high-level abstractions for GPU programming similar to the C++ Standard Template Library (STL), 
making it easier to write efficient parallel code.*/

// Handle CUDA errors
#define CUDA_CALL(call) {                                                        \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));                    \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
}

// Handle NCCL errors
#define NCCL_CALL(call) {                                                        \
    ncclResult_t err = call;                                                     \
    if (err != ncclSuccess) {                                                    \
        fprintf(stderr, "NCCL error in file '%s' in line %i: %s.\n",             \
                __FILE__, __LINE__, ncclGetErrorString(err));                    \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
}

void generate_data(std::vector<int>& A, std::vector<float>& B) {
    // Generate the sample data
    A = {49, 92, 14, 71, 60, 20, 82, 36, 44, 74}; // imagine A was uniformly distributed (0,100)
    B = {0.2790412922001377, 1.0105152848065264, -0.580878134023515, -0.5251698071781478, 
         -0.5713801657541415, -0.9240828377471049, -2.6125490126936013, 0.9503696823969031, 
         0.8164450809513273, -1.523875997615861};
}

// Partition the data into buckets for bucket sort
void partition_data(const std::vector<int>& A, const std::vector<float>& B, std::vector<std::vector<int>>& partitions_A, std::vector<std::vector<float>>& partitions_B, int num_partitions) {
    //Takes the original vectors as input and a vector of vectors of data partitions
    for (int i = 0; i < A.size(); i++) {
        int bucket = floor(num_partitions * A[i] / 100.0);
        if (bucket >= num_partitions) {
            bucket = num_partitions - 1;
        }
        partitions_A[bucket].push_back(A[i]);
        partitions_B[bucket].push_back(B[i]);
    }
}

int main(int argc, char* argv[]) {
    const int num_gpus = 2;  // Number of GPUs
    const int num_elements = 10;
    const int elements_per_gpu = num_elements / num_gpus;  // Elements per GPU

    ncclComm_t comms[num_gpus]; // comms is an array of NCCL communicators, one for each GPU, used to manage communication 
    int devs[num_gpus] = {0, 1};  // GPU IDs
    cudaStream_t streams[num_gpus];

    // Initialize NCCL
    NCCL_CALL(ncclCommInitAll(comms, num_gpus, devs)); //create communicators for each GPU

    /*Allocating and initializing device buffers*/
    int** sendbuffA = (int**) malloc(num_gpus * sizeof(int*));
    int** recvbuffA = (int**) malloc(num_gpus * sizeof(int*));
    float** sendbuffB = (float**) malloc(num_gpus * sizeof(float*));
    float** recvbuffB = (float**) malloc(num_gpus * sizeof(float*));

    // Generate the sample data
    std::vector<int> A;
    std::vector<float> B;
    generate_data(A, B);

    // Partition data
    std::vector<std::vector<int>> partitions_A(num_gpus);
    std::vector<std::vector<float>> partitions_B(num_gpus);
    partition_data(A, B, partitions_A, partitions_B, num_gpus);

    // Allocate and initialize data on each GPU
    thrust::device_vector<int> device_A[num_gpus]; // an array of thrust device vectors
    thrust::device_vector<float> device_B[num_gpus];
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(devs[i]); // Set the current device to GPU i
        device_A[i] = partitions_A[i]; //transfer data to corresponding GPU device
        device_B[i] = partitions_B[i]; //Thrust takes care of the necessary memory allocations and data transfers.
        CUDA_CALL(cudaStreamCreate(&streams[i])); // cuda streams are created to manage asynchronous operations on each GPU
    }

    // Sort data locally on each GPU
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(devs[i]); // Set the current device to GPU i
        thrust::sort_by_key(thrust::device, device_A[i].begin(), device_A[i].end(), device_B[i].begin());
    }

    // Debugging output to verify the local sorting
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(devs[i]);  // Set the current device to GPU i
        std::cout << "Sorted data on GPU " << i << ":\n";
        std::cout << "A: ";
        for (int j = 0; j < device_A[i].size(); ++j) {
            std::cout << device_A[i][j] << " ";
        }
        std::cout << "\nB: ";
        for (int j = 0; j < device_B[i].size(); ++j) {
            std::cout << device_B[i][j] << " ";
        }
        std::cout << std::endl;
    }

    thrust::device_vector<int> gathered_A_0(num_elements, 0);
    thrust::device_vector<int> gathered_A_1(num_elements, 0);
    thrust::device_vector<float> gathered_B_0(num_elements, 0.0f);
    thrust::device_vector<float> gathered_B_1(num_elements, 0.0f);

    // Copy partitions to the appropriate positions within the gathered vectors
    CUDA_CALL(cudaSetDevice(devs[0]));
    CUDA_CALL(cudaMemcpy(thrust::raw_pointer_cast(gathered_A_0.data()), thrust::raw_pointer_cast(device_A[0].data()), elements_per_gpu * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(thrust::raw_pointer_cast(gathered_B_0.data()), thrust::raw_pointer_cast(device_B[0].data()), elements_per_gpu * sizeof(float), cudaMemcpyDeviceToDevice));

    CUDA_CALL(cudaSetDevice(devs[1]));
    CUDA_CALL(cudaMemcpy(thrust::raw_pointer_cast(gathered_A_1.data()) + elements_per_gpu, thrust::raw_pointer_cast(device_A[1].data()), elements_per_gpu * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(thrust::raw_pointer_cast(gathered_B_1.data()) + elements_per_gpu, thrust::raw_pointer_cast(device_B[1].data()), elements_per_gpu * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int i = 0; i < num_gpus; ++i) {

        cudaSetDevice(i);

        cudaMalloc(&sendbuffA[i],  num_elements * sizeof(int));
        cudaMalloc(&recvbuffA[i],  num_elements * sizeof(int));
        cudaMalloc(&sendbuffB[i],  num_elements * sizeof(float));
        cudaMalloc(&recvbuffB[i],  num_elements * sizeof(float));

            switch(i) { /*Copy from host to devices*/
                case 0 : 
                    cudaMemcpy(sendbuffA[i] , thrust::raw_pointer_cast(gathered_A_0.data()),   num_elements * sizeof(int), cudaMemcpyDeviceToDevice); 
                    cudaMemcpy(sendbuffB[i] , thrust::raw_pointer_cast(gathered_B_0.data()),   num_elements * sizeof(int), cudaMemcpyDeviceToDevice); 
                    break; 
                case 1 : 
                    cudaMemcpy(sendbuffA[i] , thrust::raw_pointer_cast(gathered_A_1.data()),   num_elements * sizeof(float), cudaMemcpyDeviceToDevice); 
                    cudaMemcpy(sendbuffB[i] , thrust::raw_pointer_cast(gathered_B_1.data()),   num_elements * sizeof(float), cudaMemcpyDeviceToDevice); 
                    break;  
            }

    } 

    ncclGroupStart();
        for(int g = 0; g < num_gpus; g++) {
   	      cudaSetDevice(g);
          ncclAllGather(sendbuffA[g] + g * elements_per_gpu, recvbuffA[g], elements_per_gpu, ncclInt, comms[g], streams[g]);
          ncclAllGather(sendbuffB[g] + g * elements_per_gpu, recvbuffB[g], elements_per_gpu, ncclFloat, comms[g], streams[g]); /*All Gathering the data on GPUs*/
        }

    ncclGroupEnd();

    // Synchronize the streams to ensure data is gathered
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CALL(cudaSetDevice(devs[i]));
        CUDA_CALL(cudaStreamSynchronize(streams[i]));
    }

    // Transfer data back to host for verification
    thrust::host_vector<int> host_final_sorted_A(num_elements);
    thrust::host_vector<float> host_final_sorted_B(num_elements);

    // this works for recvbuffA[0] and recvbuffA[1], both should contain the same thing, that is the point of AllGather
    CUDA_CALL(cudaMemcpy(host_final_sorted_A.data(), recvbuffA[0], num_elements * sizeof(int), cudaMemcpyDeviceToHost)); // recvbuffA[0] and recvbuffA[1]
    CUDA_CALL(cudaMemcpy(host_final_sorted_B.data(), recvbuffB[0], num_elements * sizeof(float), cudaMemcpyDeviceToHost)); // recvbuffB[0] and recvbuffB[1]

    std::cout << "Final sorted data for A:\n";
    for (int j = 0; j < host_final_sorted_A.size(); ++j) {
        std::cout << host_final_sorted_A[j] << " ";
    }
    std::cout << "\nFinal sorted data for B:\n";
    for (int j = 0; j < host_final_sorted_B.size(); ++j) {
        std::cout << host_final_sorted_B[j] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CALL(cudaFree(sendbuffA[i]));
        CUDA_CALL(cudaFree(recvbuffA[i]));
        CUDA_CALL(cudaFree(sendbuffB[i]));
        CUDA_CALL(cudaFree(recvbuffB[i]));
    }

    free(sendbuffA);
    free(recvbuffA);
    free(sendbuffB);
    free(recvbuffB);

    // Destroy CUDA streams
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CALL(cudaStreamDestroy(streams[i]));
    }

    // Finalize NCCL
    for (int i = 0; i < num_gpus; ++i) {
        NCCL_CALL(ncclCommDestroy(comms[i]));
    }

    return 0;
}



