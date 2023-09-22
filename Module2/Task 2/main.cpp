#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>
#include <climits>
#include <fstream>
#include <string>
#include "SequentialQuickSort.h"
#include "ParallelQuickSort.h"

struct taskData{
    std::string type;
    double duration;
    int size;
    bool sorted;
};

/**
 * Prints the elements of an array.
 * @param array[] The array to be printed.
 * @param sz The size of the array.
 */
void printArray(int array[], int sz)
{
    int count = 0;
    std::cout << "[";
    for(int i = 0; i < sz; i++)
    {
        if(count == 10) {std::cout << std::endl; count = 0;}
        else{ count++; }
        if(i == sz - 1) { std::cout << array[i]; }
        else
        {
            std::cout << array[i] << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

/**
 * Generates a random array of integers.
 * @param sz The size of the array.
 * @param low The lower bound of the random numbers.
 * @param high The upper bound of the random numbers.
 */
int* randomArray(int sz, int low, int high)
{
    int* arr = new int[sz];
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);

    std::uniform_int_distribution<int> dist(low, high);
#pragma omp parallel for default(none) firstprivate(sz, dist, gen)  shared(arr)
    for(int i = 0; i < sz; i++)
    {
        arr[i] = dist(gen);
    }

    return arr;

}
/**
 * Checks if an array is sorted in ascending order.
 * @param arr[] The array to be checked.
 * @param sz The size of the array.
 */
bool isSorted(int arr[], int sz)
{
    for(int i = 1; i < sz; i++)
    {
        if( arr[i] < arr[i - 1] ) { return false; }
    }
    return true;
}

/**
 * @brief Writes task data to a CSV file.
 * @param data The task data to be written to the file.
 */
void writeCSV(taskData data) {
    std::ofstream outfile("results.csv", std::ios_base::app);
    outfile << data.type << "," << data.size << "," << data.duration << "," << std::boolalpha << data.sorted << std::endl;
    outfile.close();
}

int main() {

    int max_sz = 1000*1000*10; // 10 Million
    int loopIncrement = 100;

    omp_set_num_threads(4);

    for(int sz = 1000000; sz <= max_sz; sz += loopIncrement)
    {
        // This is to ensure an even range of data is collected.
        switch(sz)
        {
            case 10000:
                loopIncrement = 1000;
                break;
            case 100000:
                loopIncrement = 10000;
                break;
            case 1000000:
                loopIncrement = 100000;
                break;
            default:
                break;
        }
        std::cout << "Sorting Size: " << sz << std::endl;


        int *arr = randomArray(sz, -1000, 1000);
        auto start = omp_get_wtime();

        SequentialQuickSort::quickSort(arr, 0, sz - 1);

        auto stop = omp_get_wtime();
        auto duration = stop - start;
        std::cout << "Time taken by Sequential function: " << duration << " seconds" << std::endl;
        bool seq = isSorted(arr, sz);
        //writeCSV(taskData{"sequential", duration, sz, seq});

        int *arr1 = randomArray(sz, -1000, 1000);

        start = omp_get_wtime();
#pragma omp parallel default(none) shared(arr1, sz)
        {
#pragma omp single
            ParallelQuickSort::quickSort(arr1, 0, sz - 1);
        }

        stop = omp_get_wtime();
        duration = stop - start;
        std::cout << "Time taken by Parallel function: " << duration << " seconds" << std::endl;
        bool par = isSorted(arr1, sz);

        //writeCSV(taskData{"parallel", duration, sz, par});



        std::cout << std::boolalpha << "Sequential Sorted: " << seq << std::endl;
        std::cout << std::boolalpha << "Parallel Sorted: " << par << std::endl;

        delete[](arr);
        delete[](arr1);
    } // End For Loop
    return 0;
}
