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
    ~Mat(){ delete[] data; }
    void create();
    void release();

    int get_width() const{ return width; }
    int get_height() const{ return height; }
    int get_channel() const {return channel; }

    const float* get_mat() const { return data; }
    int get_dims() const { return dims; }
    int get_elemsize() const { return width * height * channel; }
    std::vector<int> size() const { return {width, height, channel}; }
    void reshape(int _w);
    void reshape(int _w, int _h);
    void reshape(int _w, int _h, int _c);
    

    friend std::ostream& operator << (std::ostream& os, const Mat &m) {
        return os << "(" << m.width << ", " << m.height << ", " << m.channel << ")";
    }
    // Operators 
    void add(const Mat &m);
    void sub(const Mat &m);
    void mul(const Mat &m);

    Mat& operator= (const Mat &m){
        if (&m == this)
            return *this;
        release();

        width = m.get_width();
        height = m.get_height();
        channel = m.get_channel();
        dims = m.get_dims();
        elemsize = m.elemsize;
        
        create();
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

    // single-precision data
    float* data = nullptr;

    int width;
    int height;
    int channel;
    
    size_t elemsize; // number of elements 
    int dims;     // dimensions

    // TODO
    int blocksize;

    static int maxThreadsPerBlock;
    static int maxThreadsPerMultiProcessor;

    void copy(const Mat &m);
    void dimensionCheck(const Mat &m);
    void information(bool verbose=false);
};

// implement
inline Mat::Mat(): 
width(0), height(0), channel(0), elemsize(0), dims(0){
    information();
    create();
}
inline Mat::Mat(int _w):
width(_w), height(1), channel(1), elemsize(_w), dims(1){
    information();
    create();
}
inline Mat::Mat(int _w, int _h): 
width(_w), height(_h), channel(1), elemsize(_w * _h), dims(2){
    information();
    create();
}
inline Mat::Mat(int _w, int _h, int _c): 
width(_w), height(_h), channel(_c), elemsize(_w * _h * _c), dims(3){
    information();
    create();
}
inline Mat::Mat(const Mat& m):
width(m.get_width()), height(m.get_height()), channel(m.get_channel()), elemsize(m.elemsize), dims(m.dims){ 
    information();
    copy(m); // cuda copy
} 

inline void Mat::dimensionCheck(const Mat& m){
    if (dims != 2 || m.dims != 2){
        std::cerr << "Dimension Error: 2-dimension matrix supported only." << std::endl;
        exit(-1);
    }
    if (width != m.get_width() || height != m.get_height()){
        std::cerr << "Dimension Error: (" << width << ", " << height << ") cannot match (" << m.get_width() << ", " << m.get_height() << ")" << std::endl;
        exit(-1);
    }
}

}

#endif