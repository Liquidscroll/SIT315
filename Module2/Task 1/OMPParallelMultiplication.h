#ifndef OMP_PARALLEL_MULTIPLICATION_H
#define OMP_PARALLEL_MULTIPLICATION_H

#include <iostream>
#include <random>

namespace OMPParallelMultiplication
{
    void printMatrix(uint64_t **matrix,  uint64_t size);
    void randomMatrix(uint64_t **matrix,  uint64_t size,
                      int low, int high, int numThreads);
    void multiplyMatrix(uint64_t **m1, uint64_t **m2, uint64_t **m3, uint64_t size, int numThreads);
    uint64_t run(uint64_t size, int numThreads, int scheduleType, int chunkSize);
}


#endif
