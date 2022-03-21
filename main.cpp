#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include "cuda_utils.h"
#include "mat.h"
using namespace std;

template <class T>
void print_binary(T& num){
    unsigned char* begin = (unsigned char*) &num; // genetic pointer class
    // cannot use the void* because it cannot be reference
    size_t len = sizeof(T);
    for (size_t i = 0; i < len ; ++i){
        printf("\t%p\t %.2x\n", begin + i, begin[i]);
    }
    printf("\n");
}

int main(int argc, char** argv){
    cout << "Starting..." << endl;
    int intnum = 12345;
    float floatnum = 0.12345;
    printf("int:\n");
    print_binary(intnum);
    printf("float:\n");
    print_binary(floatnum);
    return 0;
}