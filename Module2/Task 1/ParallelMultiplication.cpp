#include "ParallelMultiplication.h"

#include <random>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>

namespace ParallelMultiplication
{
    /**
     * Print the given matrix to the console.
     * @param matrix: The matrix to be printed.
     * @param size: The size of the matrix (assumed to be square).
     */
    void printMatrix(uint64_t **matrix, const uint64_t size) {
        std::string mat;
        for(uint64_t i = 0; i < size; i++) {
            mat += "[";
            for(uint64_t j = 0; j < size; j++) {
                if(j == 0) {
                    mat += std::to_string(matrix[i][j]);
                } else {
                    mat += ", ";
                    mat += std::to_string(matrix[i][j]);
                }
            }
            mat += "]\n";
        }
        std::cout << mat << std::endl;
    }

    /**
     * Initialize the given matrix with random values.
     * @param matrix: The matrix to be initialized.
     * @param size: The size of the matrix (assumed to be square).
     * @param low: Lower bound for random values.
     * @param high: Upper bound for random values.
     */
    void randomMatrix(uint64_t **matrix, const uint64_t size, int low, int high, int numThreads)
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> dist(low, high);

        // Calculate the number of rows each thread should handle
        uint64_t rowsPerThread = size / numThreads;

        /**
         * A lambda function to populate a specified range of rows (from startRow to endRow)
         * in the matrix with random values.
         *
         * @param startRow: The starting row index (inclusive) for element population.
         * @param endRow: The ending row index (exclusive) for element population.
         */
        auto populateRows = [&](uint64_t startRow, uint64_t endRow) {
            for(uint64_t i = startRow; i < endRow; i++) {
                for(uint64_t j = 0; j < size; j++) {
                    matrix[i][j] = dist(rng);
                }
            }
        };

        std::vector<std::thread> randomThreads;
        for(int th = 0; th < numThreads; th++) {
            uint64_t startRow = th * rowsPerThread;
            // If we're at the last thread let it handle the rest
            // of the array, no matter the size.
            uint64_t endRow = (th == numThreads - 1) ? size : startRow + rowsPerThread;
            randomThreads.emplace_back(populateRows, startRow, endRow);
        }

        for(auto& t : randomThreads) { t.join(); }
    }

    /**
     * Multiply two matrices using parallel threads and store the result in a third matrix.
     * @param m1: First matrix.
     * @param m2: Second matrix.
     * @param m3: Matrix to store the result.
     * @param size: The size of the matrices (assumed to be square).
     * @param startRow: Starting row for this segment of multiplication.
     * @param endRow: Ending row for this segment of multiplication.
     */
    void multiplyMatrix(uint64_t **m1, uint64_t **m2, uint64_t **m3,
                        const uint64_t size, uint64_t startRow, uint64_t endRow)
    {
        for(uint64_t row = startRow; row < endRow; row++) {
            for(uint64_t col = 0; col < size; col++) {
                m3[row][col] = 0;
                for(uint64_t i = 0; i < size; i++) {
                    m3[row][col] += m1[row][i] * m2[i][col];
                }
            }
        }
    }

    /**
     * Initialize a 2D array of given size.
     * @param size: The size of the 2D array (assumed to be square).
     * @return Pointer to the initialized 2D array.
     */
    uint64_t **initArray(const uint64_t size)
    {
        uint64_t **arr;
        arr = new uint64_t*[size];
        for(uint64_t i = 0; i < size; i++) {
            arr[i] = new uint64_t[size];
        }
        return arr;
    }

    /**
     * Run matrix multiplication for matrices of given size using parallel threads.
     * @param size: The size of the matrices (assumed to be square).
     * @param numThreads: Number of threads to use for parallelism.
     * @return Duration taken for the multiplication operation.
     */
    uint64_t run(const uint64_t size, int numThreads) {

        uint64_t **v1, **v2, **v3;

        // Calculate number of rows per thread
        uint64_t rowsPerThread = size / numThreads;

        // Initialize matrices
        v1 = initArray(size);
        v2 = initArray(size) ;
        v3 = initArray(size) ;

        // Fill matrices with random values
        randomMatrix(v1, size, 1, 10, numThreads/2);
        randomMatrix(v2, size, 1, 10, numThreads/2);


        // Perform matrix multiplication and measure the time taken
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> multThreads;
        for(int th = 0; th < numThreads; th++) {
            uint64_t startRow = th * rowsPerThread;
            // If we're at the last thread let it handle the rest
            // of the array, no matter the size.
            uint64_t endRow = (th == numThreads - 1) ? size : startRow + rowsPerThread;
            multThreads.emplace_back(multiplyMatrix, v1, v2, v3, size, startRow, endRow);
        }

        for(auto& t : multThreads) { t.join(); }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast
                        <std::chrono::microseconds>(end - start);

        std::cout << "Parallel Multiplication took: " << duration.count() << " microseconds" << std::endl;

        // Check if the results are correct
        for(int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                int result = 0;
                for(int k = 0; k < size; k++)
                {
                    result += v1[i][k] * v2[k][j];
                }
                if(v3[i][j] != result)
                {
                    std::cout << "Error, value is not correct at: [" << i << ", " << j << "]" << std::endl;
                    std::cout << "result: " << result << ", expected: " << v3[i][j] << std::endl;

                    // Throw exception and stop program running if there's
                    // any calculation is not correct.
                    throw std::runtime_error("Result is incorrect.");
                }
            }
        }

        // Memory deallocation
        for(uint64_t i = 0; i < size; i++) {
            delete[] v1[i];
            delete[] v2[i];
            delete[] v3[i];
        }
        delete[] v1;
        delete[] v2;
        delete[] v3;

        return duration.count();
    }
};
