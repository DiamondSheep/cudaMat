#include "mat.h"

namespace cunn{

// cuda functions 

__global__ void cuda_add(float* a, const float* b, const int height){
    // Single Block
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int index = i * height + j;
    a[index] += b[index];
}

__global__ void cuda_sub(float* a, const float* b, const int height){
    // Single Block
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int index = i * height + j;
    a[index] -= b[index];
}

__global__ void cuda_mul(const float* a, const float* b, float* result, const int width, const int height){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float tmpValue = 0.0;
    for (int k = 0; k < width; ++k){
        tmpValue += a[i * height + k] * b[k * width + j];
    }
    result[i * width + j] = tmpValue;
}

// operations 

void Mat::add(const Mat& m){
    dimensionCheck(m);
    CUDA_CHECK(cudaSetDevice(DEVICE));
    size_t bytesize = sizeof(float) * height * width;
    float* d_a = NULL;
    float* d_b = NULL;
    CUDA_CHECK(cudaMalloc((void**) &d_a, bytesize));
    CUDA_CHECK(cudaMalloc((void**) &d_b, bytesize));
    CUDA_CHECK(cudaMemcpy(d_a, data, bytesize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, m.get_mat(), bytesize, cudaMemcpyHostToDevice));

    dim3 blockPerGrid (1, 1);
    dim3 threadPerGrid (width, height);
    cuda_add<<< blockPerGrid, threadPerGrid >>>(d_a, d_b, height);

    CUDA_CHECK(cudaMemcpy(data, d_a, bytesize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaDeviceReset());
}

void Mat::sub(const Mat& m){
    dimensionCheck(m);
    CUDA_CHECK(cudaSetDevice(DEVICE));
    size_t bytesize = sizeof(float) * height * width;
    float* d_a = NULL;
    float* d_b = NULL;
    CUDA_CHECK(cudaMalloc((void**) &d_a, bytesize));
    CUDA_CHECK(cudaMalloc((void**) &d_b, bytesize));
    CUDA_CHECK(cudaMemcpy(d_a, data, bytesize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, m.get_mat(), bytesize, cudaMemcpyHostToDevice));

    dim3 blockPerGrid (1, 1);
    dim3 threadPerGrid (width, height);
    cuda_sub<<< blockPerGrid, threadPerGrid >>>(d_a, d_b, height);

    CUDA_CHECK(cudaMemcpy(data, d_a, bytesize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaDeviceReset());
}

void Mat::mul(const Mat &m){
    if (dims != 2 || m.dims != 2){
        std::cerr << "Dimension Error: 2-dimension matrix supported only" << std::endl;
        exit(-1);
    }
    // dimesion check
    if (width != m.get_height() || height != m.get_width()){
        std::cerr << "Dimension Error: (" << width << ", " << height << ") cannot match (" << m.get_width() << ", " << m.get_height() << ")" << std::endl;
        exit(-1);
    }
    float* result = new float[width * width];
    CUDA_CHECK(cudaSetDevice(DEVICE));
    size_t bytesize = sizeof(float) * height * width;
    size_t bytesize_result = sizeof(float) * width * width;
    float* d_a = NULL;
    float* d_b = NULL;
    float* d_result = NULL;
    CUDA_CHECK(cudaMalloc((void **) &d_a, bytesize));
    CUDA_CHECK(cudaMalloc((void **) &d_b, bytesize));
    CUDA_CHECK(cudaMalloc((void **) &d_result, bytesize_result));

    CUDA_CHECK(cudaMemcpy(d_a, data, bytesize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, m.get_mat(), bytesize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result, result, bytesize_result, cudaMemcpyHostToDevice));

    dim3 blockPerGrid (1, 1);
    dim3 threadPerBlock (width, width);
    cuda_mul<<<blockPerGrid, threadPerBlock>>>(d_a, d_b, d_result, width, height);

    CUDA_CHECK(cudaMemcpy(result, d_result, bytesize_result, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
    delete[] data;
    data = result;
    height = width;
}

void Mat::copy(const Mat &m){
    dimensionCheck(m);
    data = new float[width * height];
    CUDA_CHECK(cudaSetDevice(DEVICE));
    size_t bytesize = sizeof(float) * height * width;
    float* d_tmp = NULL;
    CUDA_CHECK(cudaMalloc((void**) &d_tmp, bytesize));
    CUDA_CHECK(cudaMemcpy(d_tmp, m.get_mat(), bytesize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data, d_tmp, bytesize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tmp));
}

void Mat::information(bool verbose){
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, DEVICE));
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    if (verbose){
        printf("Name: %s\n", prop.name);
        printf("sharedMemPerBlock: %lu kB\n", prop.sharedMemPerBlock / 1024lu); // in byte
        printf("warpSize: %d threads\n", prop.warpSize);
        printf("maxThreadsPerBlock: %d\n", maxThreadsPerBlock);
        printf("maxThreadsPerMultiProcessor: %d\n", maxThreadsPerMultiProcessor);
        printf("maxThreadsDim: %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }
}

void Mat::create(){
    data = new float[width * height * channel];  
    for (size_t i = 0; i < elemsize; ++i){
        data[i] = 0.0f;
    }
}
void Mat::release(){
    if (data != nullptr){
        delete[] data;
    }
}
void Mat::reshape(int _w){
    if (elemsize < _w){}
}

} // namespace