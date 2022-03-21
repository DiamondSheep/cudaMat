#ifndef __MAT_H__
#define __MAT_H__
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <vector>
#include "cuda_utils.h"

namespace cunn{
const int DEVICE = 0;
__global__ void cuda_add(float* a, const float* b, const int height);

class Mat{
public:
    Mat();
    Mat(int _w);
    Mat(int _w, int _h);
    Mat(int _w, int _h, int _c);
    Mat(const Mat& _m);
    ~Mat(){ delete[] matrix; }
    void display() const;

    int get_width() const{ return width; }
    int get_height() const{ return height; }
    int get_channel() const {return channel; }

    const float* get_mat() const { return matrix; }
    std::vector<int> size() const { return {width, height, channel}; }
    int get_elemsize() const { return width * height * channel; }
    int get_dims() const { }

    friend std::ostream& operator << (std::ostream& os, const Mat &m) {
        return os << "Matrix of " << m.width << " width, " << m.height << " height.";
    }
    // Operators 
    void add(const Mat &m);
    void sub(const Mat &m);
    void mul(const Mat &m);

    Mat& operator= (const Mat &m){
        if (&m == this)
            return *this;
        delete[] matrix; 
        width = m.get_width();
        height = m.get_height();
        matrix = new float[width * height];
        std::cout << width << ", " << height << std::endl;
        copy(m);
        return (*this);
    }
    Mat& operator+= (const Mat &m){
        this->add(m);
        return *this;
    }
    Mat operator+ (const Mat &m) const{
        Mat temp(*this);
        temp += m;
        return temp;
    }
    Mat& operator-= (const Mat &m){
        this->sub(m);
        return (*this);
    }
    Mat operator- (const Mat &m) const {
        Mat temp(*this);
        temp -= m;
        return temp;
    }
    Mat& operator*= (const Mat &m){
        this->mul(m);
        return (*this);
    }
    Mat operator* (const Mat &m){
        Mat temp(*this);
        temp *= m;
        return temp;
    }

    float* matrix;

    int width;
    int height;
    int channel;
    
    int elemsize; // number of elements 
    int dims;     // dimensions

    int blocksize;
    int maxThreadsPerBlock;
    int maxBlocksPerMultiProcessor;

    void copy(const Mat &m);
    void dimensionCheck(const Mat &m){
        if (width != m.get_width() || height != m.get_height()){
            std::cerr << "Matrix dimension error: (" << width << ", " << height << ") cannot match (" << m.get_width() << ", " << m.get_height() << ")" << std::endl;
            exit(-1);
        }
    }
    void information(bool verbose=false);
};

// implement
Mat::Mat(): 
width(0), height(0), channel(0), elemsize(0), dims(0){
    information();
    matrix = new float[0];
}
Mat::Mat(int _w):
width(_w), height(0), channel(0), elemsize(0), dims(0){
    exit(-1);
}
Mat::Mat(int _w, int _h): 
width(_w), height(_h), channel(0), elemsize(_w * _h), dims(2){
    information();
    matrix = new float[width * height];
    for (int i = 0; i < width; i++){
        for (int j = 0; j < height; j++){
            matrix[i * height + j] = i+1;
        }
    }
}
Mat::Mat(int _w, int _h, int _c): 
width(_w), height(_h), channel(_c), elemsize(_w * _h * _c), dims(3){
    exit(-1);
}
Mat::Mat(const Mat& m){ 
    information();
    width = m.get_width();
    height = m.get_height();
    copy(m); // cuda copy
} 
void Mat::display() const{
    for (int i = 0; i < width; i++){
        for (int j = 0; j < height; j++){
            printf("%f\t", matrix[i * height + j]);
        }
        printf("\n");
    }
}

}

#endif