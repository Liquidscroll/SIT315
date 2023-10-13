#include <iostream>
#include <vector>
#include <stack>
#include <chrono>
#include <random>
#include <algorithm>
#include <queue>
#include <mpi.h>
#include <fstream>

std::chrono::high_resolution_clock::time_point start;
std::chrono::microseconds duration;

/**
 * A comparator struct used for comparing pairs in a priority queue.
 */
struct Compare {
    /**
     * Compares two pairs based on their first element. Returns true if the
     * first element of a is greater than the first element of b, else false.
     * @param a: The first pair.
     * @param b: The second pair.
     * @return
     */
    bool operator()(std::pair<int, std::pair<int, int>> a, std::pair<int, std::pair<int, int>> b) {
        return a.first > b.first;
    }
};

void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

/**
 * Finds the median of three elements in an array segment, and ensures they are ordered in the array.
 * @param arr: The array in which the elements reside.
 * @param low: The lower bound index of the array segment.
 * @param high: The upper bound index of the array segment.
 * @return Returns the median value of the three elements.
 */
int medianOfThree(std::vector<int> &arr, int low, int high)
{
    int mid = (low + high) / 2;
    if(arr[low] > arr[high]) {
        swap(arr[low], arr[high]);
    }
    if(arr[low] > arr[mid]) {
        swap(arr[low], arr[mid]);
    }
    if(arr[mid] < arr[high]) {
        swap(arr[mid], arr[high]);
    }
    return arr[high];
}

/**
 * Partitions the segment of the array and reorders elements based on the pivot,
 * used in QuickSort.
 * @param arr: The array to be partitioned.
 * @param low: The lower bound index of the array segment.
 * @param high: The upper bound index of the array segment.
 * @return Returns the index of the pivot element.
 */
int partition(std::vector<int> &arr, int low, int high)
{
    int pivot = medianOfThree(arr, low, high);
    int i = (low - 1);

    for(int j = low; j <= high - 1; j++)
    {
        if(arr[j] <= pivot)
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

/**
 * An iterative version of QuickSort, which sorts the array segment using a stack.
 * @param arr: The array to be sorted.
 * @param low: The lower bound index for sorting.
 * @param high: The upper bound index for sorting.
 */
void quickSortIterative(std::vector<int> &arr, int low, int high)
{
    // A stack used to store low and high indexes during the sorting process
    std::stack<int> stack;

    // Push initial low and high indexes to the stack
    stack.push(low);
    stack.push(high);

    while(!stack.empty())
    {
        high = stack.top();
        stack.pop();
        low = stack.top();
        stack.pop();

        int pivot = partition(arr, low, high);

        // If there are elements on the left side of the pivot, push left segment indices to stack
        if(pivot - 1 > low)
        {
            stack.push(low);
            stack.push(pivot - 1);
        }
        // If there are elements on the right side of the pivot, push right segment indices to stack
        if(pivot + 1 < high)
        {
            stack.push(pivot + 1);
            stack.push(high);
        }
    }
}

/**
 * Print the given matrix to the console.
 * @param matrix: The matrix to be printed.
 * @param rows: of the matrix.
 * @param cols: of the matrix.
 */
void printVector(const std::vector<int> &vec)
{
    std::cout << "[";
    for(int i = 0; i < vec.size(); i++)
    {
        if(i != 0 && i % 10 == 0) { std::cout << std::endl; }
        std::cout << vec.at(i);
        if(i != vec.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

/**
 * Initialize the given matrix with random values.
 * @param matrix: The matrix to be initialized.
 * @param rows: of the matrix.
 * @param cols: of the matrix.
 * @param low: Lower bound for random values.
 * @param high: Upper bound for random values.
 */
void randomVector(std::vector<int> &vec, int low, int high) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(low, high);

    for (int &i : vec) {
        i = dist(rng);
    }
}

/**
 * Check if a given vector is sorted.
 * @param arr: Vector to be checked.
 */
bool isSorted(const std::vector <int> &arr)
{
    for(int i = 0; i < arr.size() - 1; i++)
    {
        if(arr[i] > arr[i + 1])
        {
            return false;
        }
    }
    return true;
}

/**
 * Merges k sorted arrays into a single sorted array.
 * @param largeArray: A large array containing all the elements of k sorted arrays concatenated together.
 * @param displacements: A vector containing the starting index of each of the k sorted arrays in the large array.
 */
std::vector<int> mergeKSortedArrays(std::vector<int>& largeArray, std::vector<int>& displacements) {
    std::vector<std::vector<int>> arrays;
    for(int i = 0; i < displacements.size(); i++)
    {

        int startIdx = displacements[i];
        int endIdx = (i == displacements.size() - 1) ? largeArray.size() : displacements[i + 1];

        arrays.emplace_back(largeArray.begin() + startIdx, largeArray.begin() + endIdx);
    }


    std::vector<int> result;
    std::priority_queue<std::pair<int, std::pair<int, int>>,
            std::vector<std::pair<int, std::pair<int, int>>>, Compare> min_heap;

    // Insert the first element of each array into the heap
    for (int i = 0; i < arrays.size(); i++) {
        if (!arrays[i].empty()) {
            min_heap.push({arrays[i][0], {i, 0}});
        }
    }

    while (!min_heap.empty()) {
        std::pair<int, std::pair<int, int>> current = min_heap.top();
        min_heap.pop();

        int value = current.first;
        int arrayIndex = current.second.first;
        int elementIndex = current.second.second;

        result.push_back(value);

        // If there's another element in the same array, push it into the heap
        if (elementIndex + 1 < arrays[arrayIndex].size()) {
            min_heap.push({arrays[arrayIndex][elementIndex + 1], {arrayIndex, elementIndex + 1}});
        }
    }
    return result;
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
                double time,
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
         << time << ","
         << (sorted ? "true" : "false") << "\n";

    // Close the file
    file.close();
    return true;
}

int main(int argc, char* argv[])
{
    int size = 1000 * 10; // Default: 10,000

    MPI_Init(&argc, &argv);

    int worldRank, worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);


    std::vector<int> arr(size);
    if(worldRank == 0)
    {
        randomVector(arr, 0, 10);
    }

    int localSize = size / worldSize;
    int rem = size % worldSize;
    int sum = 0;

    std::vector<int> sendCounts(worldSize), displs(worldSize);
    // Determine how much data each process will be sent/receive.
    for (int i = 0; i < worldSize; i++) {
        sendCounts[i] = (localSize + (i < rem ? 1 : 0));
        displs[i] = sum;
        sum += localSize + (i < rem ? 1 : 0);
    }

    std::vector<int> localArr(localSize);
    if(worldRank == 0)
    {
        start = std::chrono::high_resolution_clock::now();
    }

    // Distribute the array among all processes using MPI_Scatterv
    MPI_Scatterv(arr.data(), sendCounts.data(), displs.data(), MPI_INT,
                 localArr.data(),sendCounts[worldRank], MPI_INT, 0, MPI_COMM_WORLD);

    quickSortIterative(localArr, 0, localArr.size() - 1);

    // Gather the sorted subarrays from all processes back to the root process.
    MPI_Gatherv(localArr.data(), sendCounts[worldRank], MPI_INT, arr.data(),
                sendCounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    // If the current process is the root, merge the sorted subarrays and stop the clock.
    if(worldRank == 0)
    {
        arr = mergeKSortedArrays(arr, displs);

        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time Taken my MPI Quicksort: " << duration.count() << " microseconds" << std::endl;

        //std::cout << "Sorted Array: " << std::endl;
        //printVector(arr);

        bool sorted = isSorted(arr);
        if(!sorted)
        {
            std::cout << "Is Sorted: " << std::boolalpha << sorted << std::endl;
        }
        writeToCSV("output.csv", "MPI_Quicksort", size, duration.count(), sorted);
    }


    MPI_Finalize();
    return 0;

}