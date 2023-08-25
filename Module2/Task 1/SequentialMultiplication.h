#ifndef SEQUENTIAL_MULTIPLICATION_H
#define SEQUENTIAL_MULTIPLICATION_H

#include <iostream>
#include <random>

namespace SequentialMultiplication
{
    void printMatrix(int **matrix,  uint64_t size);
    void randomMatrix(int **matrix,  uint64_t size, std::mt19937 &rng,
                      int low, int high);
    uint64_t run(uint64_t size);

}


#endif
