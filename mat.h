#ifndef __MAT_H__
#define __MAT_H__
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "cuda_utils.h"

const int DEVICE = 0;
__global__ void cuda_add(float* a, const float* b, const int column);

class Mat{
public:
    Mat(): row(0), column(0){
        matrix = new float[0];
    }
    Mat(int m, int n): row(m), column(n){
        matrix = new float[row * column];
        for (int i = 0; i < row; i++){
            for (int j = 0; j < column; j++){
                matrix[i * column + j] = 1.0;
            }
        }
    }
    Mat(const Mat& m){ 
        row = m.get_row();
        column = m.get_column();
        copy(m); } // cuda copy
    ~Mat(){ delete[] matrix; }
    void display() const{
        for (int i = 0; i < row; i++){
            for (int j = 0; j < column; j++){
                printf("%f\t", matrix[i * column + j]);
            }
            printf("\n");
        }
    }

    int get_row() const{ return row; }
    int get_column() const{ return column; }
    const float* get_mat() const { return matrix; }

    friend std::ostream& operator << (std::ostream& os, const Mat &m) {
        return os << "Matrix of " << m.row << " row, " << m.column << " column.";
    }
    // Operators 
    void add(const Mat &m);
    void sub(const Mat &m);
    Mat mul(const Mat &m);
    Mat& operator= (const Mat &m){
        if (&m == this)
            return *this;
        delete[] matrix; 
        row = m.get_row();
        column = m.get_column();
        matrix = new float[row * column];
        std::cout << row << ", " << column << std::endl;
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

private:
    float* matrix;
    int row;
    int column;
    void copy(const Mat &m);
    void dimensionCheck(const Mat &m){
        if (row != m.get_row() || column != m.get_column()){
            std::cerr << "Matrix dimension error: (" << row << ", " << column << ") cannot match (" << m.get_row() << ", " << m.get_column() << ")" << std::endl;
            exit(-1);
        }
    }
};


#endif