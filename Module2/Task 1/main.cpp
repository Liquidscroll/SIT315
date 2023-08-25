#include "SequentialMultiplication.h"
#include "ParallelMultiplication.h"
#include "OMPParallelMultiplication.h"

#include <iostream>
#include <random>
#include <fstream>
#include <thread>

/**
 * Structure to hold the results of matrix multiplication tests.
 */
struct testResults
{
    std::string type;
    uint64_t numThreads{};
    uint64_t chunkSize{};
    uint64_t time{};
    uint64_t size{};

};

/**
 * Print the test results to the console.
 * @param tr: The test results to be printed.
 */
void printTestResults(const testResults& tr)
{
    std::cout << tr.type << " | " << std::to_string(tr.numThreads) << " | " << std::to_string(tr.time)
                << " | " << std::to_string(tr.size) << std::endl;
}

/**
 * Write the test results to a CSV file.
 * @param filename: Name of the CSV file to write to.
 * @param data: Vector containing the test results.
 */
void writeCSV(const std::string& filename, const std::vector<testResults>& data) {
    // Open the CSV file for writing
    std::ofstream csvFile(filename);

    // Check if the file is opened successfully
    if (!csvFile.is_open()) {
        std::cerr << "Failed to open the CSV file for writing." << std::endl;
        return;
    }
    // Write headers to the CSV file

    csvFile << "type,numThreads,chunksize,time,size" << std::endl;
    // Write the data to the CSV file
    for (const auto& row : data) {
        csvFile << row.type << ","
                << row.numThreads << ","
                << row.chunkSize << ","
                << row.time << ","
                << row.size << std::endl;
    }

    // Close the CSV file
    csvFile.close();
}

int main() {

    uint64_t size = 1000;

    // Get maxThreads for current hardware.
    unsigned int maxThreads = std::thread::hardware_concurrency();
    std::vector<testResults> results;

    // Test matrix multiplication for different matrix sizes
    for(uint64_t minSize = size; minSize > 0; minSize -= 100) {
        std::cout << "Testing size: " << minSize << std::endl;

        // Sequential test
        testResults seq;
        seq.type = "Sequential";
        seq.numThreads = 1;
        seq.time = SequentialMultiplication::run(minSize);
        seq.size = minSize;
        seq.chunkSize = minSize;
        results.push_back(seq);

        // Parallel tests for different thread counts
        for (unsigned int th = 2; th <= maxThreads; th++) {
            std::cout << "Testing Threads: " << th << std::endl << std::endl;

            testResults par;
            par.type = "Parallel";
            par.numThreads = th;
            par.time = ParallelMultiplication::run(minSize, th);
            par.size = minSize;
            seq.chunkSize = minSize / th;
            results.push_back(par);

            // Iterate over and test different scheduling
            // types - Auto, Static, Dynamic, Guided
            for(int i = 0; i < 4; i++)
            {
                testResults omp;
                omp.type = "OMP";
                omp.numThreads = th;
                omp.size = minSize;
                switch(i)
                {
                    case 0:
                        omp.type += "_AUTO";
                        omp.chunkSize = -1;
                        omp.time = OMPParallelMultiplication::run(minSize, th, i, -1);
                        results.push_back(omp);
                        break;
                    case 1:
                        omp.type += "_STATIC";
                        // Iterate over different chunk sizes in 100 increments
                        // if applicable to the type of schedule.
                        for(int chunkSize = minSize; chunkSize >= 0; (chunkSize % 100 == 0) ? chunkSize -= 100 : chunkSize--)
                        {
                            omp.chunkSize = chunkSize;
                            omp.time = OMPParallelMultiplication::run(minSize, th, i, chunkSize);
                            results.push_back(omp);
                        }
                        break;
                    case 2:
                        omp.type += "_DYNAMIC";
                        // Iterate over different chunk sizes in 100 increments
                        // if applicable to the type of schedule.
                        for(int chunkSize = minSize; chunkSize >= 1; (chunkSize % 100 == 0) ? chunkSize -= 100 : chunkSize--)
                        {
                            omp.chunkSize = chunkSize;
                            omp.time = OMPParallelMultiplication::run(minSize, th, i, chunkSize);
                            results.push_back(omp);
                        }
                        break;
                    case 3:
                        omp.type += "_GUIDED";
                        // Iterate over different chunk sizes in 100 increments
                        // if applicable to the type of schedule.
                        for(int chunkSize = minSize; chunkSize >= 0; (chunkSize % 100 == 0) ? chunkSize -= 100 : chunkSize--)
                        {
                            omp.chunkSize = chunkSize;
                            omp.time = OMPParallelMultiplication::run(minSize, th, i, chunkSize);
                            results.push_back(omp);
                        }
                        break;

                    default:
                        // If we're here something has gone very wrong.
                        throw new std::exception();
                }
            }
        }
    }

    writeCSV("output_new.csv", results);

    return 0;
}
