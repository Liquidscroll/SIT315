//
// Created by jojat on 2/10/2023.
//



#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <fstream>
#include <iostream>
#include <vector>
#include <mpi.h>
#include <omp.h>



/**
 * Print the given matrix to the console.
 * @param matrix: The matrix to be printed.
 * @param rows: of the matrix.
 * @param cols: of the matrix.
 */
void printMatrix(std::vector<int> matrix, int rows, int cols) {
    std::string mat;
    for(uint64_t i = 0; i < rows; i++) {
        mat += "[";
        for(uint64_t j = 0; j < cols; j++) {
            if(j == 0) {
                mat += std::to_string(matrix[i * rows + j]);
            } else {
                mat += ", ";
                mat += std::to_string(matrix[i * rows + j]);
            }
        }
        mat += "]\n";
    }
    std::cout << mat << std::endl;
}

/**
 * Initialize the given matrix with random values.
 * @param matrix: The matrix to be initialized.
 * @param rows: of the matrix.
 * @param cols: of the matrix.
 * @param low: Lower bound for random values.
 * @param high: Upper bound for random values.
 */
void randomMatrix(std::vector<int> &matrix, int rows, int cols, int low, int high)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(low, high);

    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = dist(rng);
        }
    }
}

/**
 * Initialize a 1D vector for matrix of given rows and cols.
 * @param rows: of the matrix.
 * @param cols: of the matrix.
 */
std::vector<int> initArray(const int rows, const int cols)
{
    std::vector<int> arr(rows * cols);
    return arr;
}

/**
 * Function to write data to CSV.
 * @param filename:  Name of file to write to.
 * @param type: Type of Matrix Multiplication.
 * @param size: Size of matrix (assumed to be square: size x size)
 * @param duration: Duration of multiplication (microseconds)
 * @param sorted : If the matrix was sorted correctly.
 */
bool writeToCSV(const std::string& filename,
                const std::string& type,
                int size,
                double duration,
                bool sorted) {

    // Open the file for writing
    std::ofstream file(filename, std::ios::app); // std::ios::app to append to the file

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Could not open the file " << filename << std::endl;
        return false;
    }

    // Check if the file is empty and if so, write the headers
    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "type,size,duration,sorted\n";
    }

    // Write the data to the file
    file << type << ","
         << size << ","
         << duration << ","
         << (sorted ? "true" : "false") << "\n";

    // Close the file
    file.close();
    return true;
}
/**
 * Main function to multiple matrices using MPI with OMP on the nodes.
 */
int main(int argc, char **argv)
{

    int size = 4;
    MPI_Init(&argc, &argv);
    if(argc > 1)
    {
        size = std::stoi(argv[1]);
    }

    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::vector<int> sendCounts(worldSize), displs(worldSize);
    std::vector<int> v1, v2, v3;

    // Generate data on root process.
    if(worldRank == 0) {
        v1 = initArray(size, size);
        v2 = initArray(size, size);
        v3 = initArray(size, size);
        randomMatrix(v1, size, size, 0, 10);
        randomMatrix(v2, size, size, 0, 10);
        //printMatrix(v1, size, size);
        //printMatrix(v2, size, size);
    }else {
        // Resize v2 and v3 in non-root processes to avoid null pointers
        v2.resize(size * size);
        v3.resize(size * size);
    }

    int rowsPerProcess = size / worldSize;
    int rem = size % worldSize;
    int sum = 0;

    // Calculate the number of elements to be sent to each process and the associated displacement within the array.
    // If there is a remainder when dividing size by processes, then we can distribute an additional row to each
    // process, who's rank is below the remainder.
    for (int i = 0; i < worldSize; i++) {
        sendCounts[i] = (rowsPerProcess + (i < rem ? 1 : 0)) * size;
        displs[i] = sum * size;
        sum += rowsPerProcess + (i < rem ? 1 : 0);
    }

    // Each process will receive a number of elements in v1 and the same number to v3 (the results matrix.)
    // So we create sub-arrays to hold these elements.
    std::vector<int> v1_sub(sendCounts[worldRank]),
                     v3_sub(sendCounts[worldRank]);

    std::chrono::high_resolution_clock::time_point start;
    if(worldRank == 0) { start = std::chrono::high_resolution_clock::now(); }

    // Scatter the data to each process. We need to give MPI a pointer to beginning of the data buffers, and we do
    // this by using .data(). We use MPI_Scatterv as each process will receive a different number of elements.
    MPI_Scatterv(v1.data(), sendCounts.data(), displs.data(), MPI_INT,
                 v1_sub.data(), sendCounts[worldRank], MPI_INT, 0, MPI_COMM_WORLD);
    // v2 will be used by all processes, so we need to broadcast it.
    MPI_Bcast(v2.data(), size * size, MPI_INT, 0, MPI_COMM_WORLD);

    // We use OMP here to speed up the multiplication. We use a number of threads equal to world size, and we also
    // use collapse to make the 2 outer-loops one.
#pragma omp parallel for default(none) shared(v1_sub, v2, v3_sub, sum) firstprivate(size, sendCounts, worldRank) num_threads(worldSize) collapse(2) schedule(auto)
    for(int i = 0; i < sendCounts[worldRank] / size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            sum = 0;
            // We then use a parallel reduction section on sum to calculate the work, and then assign it to the
            // appropriate position in the results.
#pragma omp parallel reduction(+:sum)
            for(int k = 0; k < size; k++)
            {
                sum += v1_sub[i * size + k] * v2[k * size + j];
            }
            v3_sub[i * size + j] = sum;
        }
    }

    // We then receive the results, using MPI_Gatherv as we have a variable number of elements to receive into v3.
    MPI_Gatherv(v3_sub.data(), sendCounts[worldRank], MPI_INT, v3.data(),
                sendCounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);


    if(worldRank == 0) {
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast
                <std::chrono::microseconds>(end - start);

        std::cout << "OMP + MPI Multiplication took: " << duration.count() << " microseconds" << std::endl;

        bool sorted = true;
        //printMatrix(v1, size, size);
        //printMatrix(v2, size, size);
        //printMatrix(v3, size, size);
        for(int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int result = 0;
                for(int k = 0; k < size; k++) {
                    result += v1[i * size + k] * v2[k * size + j];
                }
                if(v3[i * size + j] != result) {
                    sorted = false;
                    std::cout << "Error, value is not correct at: [" << i << ", " << j << "]" << std::endl;
                    std::cout << "result: " << result << ", expected: " << v3[i * size + j] << std::endl;

                    // Throw exception and stop program running if there's
                    // any calculation is not correct.
                }
            }
        }
        writeToCSV("omp_mpi_results.csv", "omp_mpi", size, duration.count(), sorted);

    }
    MPI_Finalize();
    return 0;
}
