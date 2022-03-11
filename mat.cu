#include "mat.h"

__global__ void cuda_add(float* a, const float* b, const int column){
    // Single Block
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int index = i * column + j;
    a[index] += b[index];
}

__global__ void cuda_sub(float* a, const float* b, const int column){
    // Single Block
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int index = i * column + j;
    a[index] -= b[index];
}

__global__ void cuda_mul(const float* a, const float* b, float* result, const int row, const int column){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float tmpValue = 0.0;
    for (int k = 0; k < row; ++k){
        tmpValue += a[i * column + k] * b[k * row + j];
    }
    result[i * row + j] = tmpValue;
}

__global__ void cuda_mul_shared(const float* a, const float* b, float* result, const int row, const int column)
{
    __shared__ float Mds[1][1];
    __shared__ float Nds[1][1];

}

void Mat::add(const Mat& m){
    dimensionCheck(m);
    CUDA_CHECK(cudaSetDevice(DEVICE));
    size_t bytesize = sizeof(float) * column * row;
    float* d_a = NULL;
    float* d_b = NULL;
    CUDA_CHECK(cudaMalloc((void**) &d_a, bytesize));
    CUDA_CHECK(cudaMalloc((void**) &d_b, bytesize));
    CUDA_CHECK(cudaMemcpy(d_a, matrix, bytesize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, m.get_mat(), bytesize, cudaMemcpyHostToDevice));

    dim3 blockPerGrid (1, 1);
    dim3 threadPerGrid (row, column);
    cuda_add<<< blockPerGrid, threadPerGrid >>>(d_a, d_b, column);

    CUDA_CHECK(cudaMemcpy(matrix, d_a, bytesize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaDeviceReset());
}

void Mat::sub(const Mat& m){
    dimensionCheck(m);
    CUDA_CHECK(cudaSetDevice(DEVICE));
    size_t bytesize = sizeof(float) * column * row;
    float* d_a = NULL;
    float* d_b = NULL;
    CUDA_CHECK(cudaMalloc((void**) &d_a, bytesize));
    CUDA_CHECK(cudaMalloc((void**) &d_b, bytesize));
    CUDA_CHECK(cudaMemcpy(d_a, matrix, bytesize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, m.get_mat(), bytesize, cudaMemcpyHostToDevice));

    dim3 blockPerGrid (1, 1);
    dim3 threadPerGrid (row, column);
    cuda_sub<<< blockPerGrid, threadPerGrid >>>(d_a, d_b, column);

    CUDA_CHECK(cudaMemcpy(matrix, d_a, bytesize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaDeviceReset());
}

void Mat::mul(const Mat &m){
    // dimesion check
    if (row != m.get_column() || column != m.get_row()){
        std::cerr << "Matrix dimension error: (" << row << ", " << column << ") cannot match (" << m.get_row() << ", " << m.get_column() << ")" << std::endl;
        exit(-1);
    }
    float* result = new float[row * row];
    CUDA_CHECK(cudaSetDevice(DEVICE));
    size_t bytesize = sizeof(float) * column * row;
    size_t bytesize_result = sizeof(float) * row * row;
    float* d_a = NULL;
    float* d_b = NULL;
    float* d_result = NULL;
    CUDA_CHECK(cudaMalloc((void **) &d_a, bytesize));
    CUDA_CHECK(cudaMalloc((void **) &d_b, bytesize));
    CUDA_CHECK(cudaMalloc((void **) &d_result, bytesize_result));

    CUDA_CHECK(cudaMemcpy(d_a, matrix, bytesize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, m.get_mat(), bytesize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result, result, bytesize_result, cudaMemcpyHostToDevice));

    dim3 blockPerGrid (1, 1);
    dim3 threadPerBlock (row, row);
    cuda_mul<<<blockPerGrid, threadPerBlock>>>(d_a, d_b, d_result, row, column);

    CUDA_CHECK(cudaMemcpy(result, d_result, bytesize_result, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
    delete[] matrix;
    matrix = result;
    column = row;
}

void Mat::copy(const Mat &m){
    dimensionCheck(m);
    matrix = new float[row * column];
    CUDA_CHECK(cudaSetDevice(DEVICE));
    size_t bytesize = sizeof(float) * column * row;
    float* d_tmp = NULL;
    CUDA_CHECK(cudaMalloc((void**) &d_tmp, bytesize));
    CUDA_CHECK(cudaMemcpy(d_tmp, m.get_mat(), bytesize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrix, d_tmp, bytesize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tmp));
}

void Mat::information(bool verbose){
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, DEVICE));
    maxThreadsPerBlock = prop.maxThreadsPerBlock;
    maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    if (verbose){
        printf("Name: %s\n", prop.name);
        printf("sharedMemPerBlock: %lu kB\n", prop.sharedMemPerBlock / 1024lu); // in byte
        printf("warpSize: %d threads\n", prop.warpSize);
        printf("maxThreadsPerBlock: %d\n", maxThreadsPerBlock);
        printf("maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("maxThreadsDim: %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }
}
    