#include <iostream>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/join.hpp>
#include <cuda_runtime.h>

// Handle CUDA errors
#define CUDA_CALL(call) {                                                        \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));                    \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
}

int main(int argc, char* argv[]) {
    // Read data from CSV files into cuDF tables
    cudf::io::csv_reader_options data1_options = cudf::io::csv_reader_options::builder("/Users/mihirsavkar/Desktop/NCCL/data1.csv");
    auto table_A = cudf::io::read_csv(data1_options);

    cudf::io::csv_reader_options data2_options = cudf::io::csv_reader_options::builder("/Users/mihirsavkar/Desktop/NCCL/data2.csv");
    auto table_B = cudf::io::read_csv(data2_options);

    // Perform the join operation
    cudf::table result = cudf::inner_join(table_A->view(), table_B->view(), {"id"}, {"id"}, {});

    // Print the result
    std::cout << result << std::endl;

    return 0;
}
