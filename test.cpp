#include <iostream>
#include "mat.h"

void test_mat(){
    using namespace cunn;
    std::cout << "-- Initial ..." << std::endl;
    Mat mat0;
    Mat mat1(1);
    Mat mat2(1, 2);
    Mat mat3(2, 2, 3);
    Mat mat4(mat1);
    
}

int main(){
    std::cout << "-- cunnlib testing..." << std::endl;
    test_mat();
    return 0;
}