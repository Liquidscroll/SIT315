//
// Created by mandi on 14/08/2023.
//

#ifndef PARALLEL_MULTIPLICATION_H
#define PARALLEL_MULTIPLICATION_H

#include <iostream>
#include <random>

namespace ParallelMultiplication
{
    void printMatrix(uint64_t **matrix,  uint64_t size);
    void randomMatrix(uint64_t **matrix,  uint64_t size,
                      int low, int high);
    void multiplyMatrix(uint64_t **m1, uint64_t **m2, uint64_t **m3, uint64_t size, uint64_t startRow, uint64_t endRow);
    uint64_t run(uint64_t size, int numThreads);

}


#endif
