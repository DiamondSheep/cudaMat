#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include "cuda_utils.h"
#include "mat.h"
using namespace std;

int main(int argc, char** argv){
    cout << "Starting..." << endl;
    Mat mat1(2, 3);
    Mat mat2(2, 3);
    Mat mat3(3, 2);
    cout << "Mat1: " << mat1 << endl;
    mat1.display();
    cout << "Mat2: " << mat2 << endl;
    mat2.display();
    cout << "Mat3: " << mat3 << endl;
    mat3.display();
    cout << " --- matrix add --- " << endl;
    mat3 *= mat1;
    cout << "Mat3: " << mat3 << endl;
    mat3.display();

    return 0;
}