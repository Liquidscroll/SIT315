#include "SequentialMultiplication.h"

#include <random>
#include <chrono>

namespace SequentialMultiplication
{
    void printMatrix(int **matrix, const uint64_t size) {
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
     * @param rng: Random number generator.
     * @param low: Lower bound for random values.
     * @param high: Upper bound for random values.
     */
    void randomMatrix(int **matrix, const uint64_t size, std::mt19937 &rng,
                      int low, int high)
    {
        std::uniform_int_distribution<int> dist(low, high);
        for(uint64_t i = 0; i < size; i++) {
            for(uint64_t j = 0; j < size; j++) {
                matrix[i][j] = dist(rng);
            }
        }
    }

    /**
     * Run matrix multiplication for matrices of given size.
     * @param size: The size of the matrices (assumed to be square).
     * @return Duration taken for the multiplication operation.
     */
    uint64_t run(const uint64_t size) {

        std:: random_device rd;
        std::mt19937 rng(rd());

        int **v1, **v2, **v3;

        // Memory allocation for matrices
        v1 = new int*[size];
        v2 = new int*[size];
        v3 = new int*[size];
        for(uint64_t i = 0; i < size; i++){
            v1[i] = new int[size];
            v2[i] = new int[size];
            v3[i] = new int[size];
        }

        // Initialize matrices with random values
        randomMatrix(v1, size, rng, 1, 10);
        randomMatrix(v2, size, rng, 1, 10);
        // Perform matrix multiplication and measure the time taken
        auto start = std::chrono::high_resolution_clock::now();

        for(uint64_t row = 0; row < size; row++) {
            for(uint64_t col = 0; col < size; col++) {
                v3[row][col] = 0;
                for(uint64_t i = 0; i < size; i++) {
                    v3[row][col] += v1[row][i] * v2[i][col];
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast
                        <std::chrono::microseconds>(end - start);

        std::cout << "Sequential Multiplication took: " << duration.count() << " microseconds" << std::endl;

        // Check if the results are correct
        for(int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int result = 0;
                for(int k = 0; k < size; k++) {
                    result += v1[i][k] * v2[k][j];
                }
                if(v3[i][j] != result) {
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
