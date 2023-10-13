/*
 * Contents of:
 *      quickSort.cl
 *
 *
 * // Swap function
 *  void swap(__global int* a, __global int* b)
 *  {
 *      int temp = *a;
 *      *a = *b;
 *      *b = temp;
 *  }
 *
 *  // Median of Three function
 *  int medianOfThree(__global int* arr, int low, int high)
 *  {
 *     int mid = (low + high) / 2;
 *
 *     if(arr[low] > arr[high]) {
 *         swap(&arr[low], &arr[high]);
 *     }
 *     if(arr[low] > arr[mid]) {
 *         swap(&arr[low], &arr[mid]);
 *     }
 *      if(arr[mid] < arr[high]) {
 *         swap(&arr[mid], &arr[high]);
 *     }
 *     return arr[high];
 *  }
 *
 *  // Partition function
 *  int partition(__global int* arr, int low, int high)
 *  {
 *      int pivot = medianOfThree(arr, low, high);
 *      int i = (low - 1);
 *
 *      for(int j = low; j <= high - 1; j++)
 *      {
 *          if(arr[j] <= pivot)
 *          {
 *              i++;
 *              swap(&arr[i], &arr[j]);
 *          }
 *      }
 *      swap(&arr[i + 1], &arr[high]);
 *      return (i + 1);
 *  }
 *
 *  __kernel void quickSort(__global int* arr, int low, int high)
 *  {
 *
 *      const int STACK_SIZE = 100000;
 *      int stack[STACK_SIZE]; // Cannot set dynamically, so we will make this a large number.
 *      int stackIndex = 0;
 *
 *      // Push initial low and high onto stack
 *      stack[stackIndex] = low;
 *      stackIndex++;
 *      stack[stackIndex] = high;
 *      stackIndex++;
 *      while(stackIndex > 0)
 *      {
 *          // Pop high and low off stack
 *          stackIndex--;
 *          high = stack[stackIndex];
 *          stackIndex--;
 *          low = stack[stackIndex];
 *
 *          int pivot = partition(arr, low, high);
 *
 *          if(pivot - 1 > low)
 *          {
 *              stack[stackIndex] = low;
 *              stackIndex++;
 *              stack[stackIndex] = pivot - 1;
 *              stackIndex++;
 *          }
 *
 *          if(pivot + 1 < high)
 *          {
 *              stack[stackIndex] = pivot + 1;
 *              stackIndex++;
 *              stack[stackIndex] = high;
 *              stackIndex++;
 *          }
 *      }
 *  }
 *
 *
 *
 *
 */


#include <iostream>
#include <vector>
#include <stack>
#include <chrono>
#include <random>
#include <algorithm>
#include <queue>
#include <mpi.h>
#include <CL/cl.h>
#include <fstream>
int SZ = 4;

// Init global OpenCL components.
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;

cl_event event = NULL;
cl_int err;

// These buffers will store the matrices being worked on.
cl_mem bufA, bufB, bufC;

std::chrono::high_resolution_clock::time_point start;
std::chrono::microseconds duration;


size_t global[1];

/**
 * Helper function to print out error messages consistently.
 * @param worldRank: Rank of calling process.
 * @param message: Error message to print.
 */
void printError(int worldRank, std::string message)
{
    printf("%s\n", message.c_str());
    printf("Rank: %d -- Error: %d\n", worldRank, err);
    exit(1);
}

/**
 * Function to release OpenCL object memory to avoid memory leaks.
 * @param rank: Rank of the current process. Used for error messaging.
 */
void free_memory(int rank)
{
    err = clReleaseKernel(kernel);
    if(err < 0) {  printError(rank, "Couldn't release kernel"); }
    err = clReleaseMemObject(bufA);
    if(err < 0) { printError(rank, "Couldn't release bufA"); }
    err = clReleaseMemObject(bufB);
    if(err < 0) { printError(rank, "Couldn't release bufB"); }
    err = clReleaseMemObject(bufC);
    if(err < 0) { printError(rank, "Couldn't release bufC"); }
    err = clReleaseCommandQueue(queue);
    if(err < 0) { printError(rank, "Couldn't release queue"); }
    err = clReleaseProgram(program);
    if(err < 0) { printError(rank, "Couldn't release program"); }
    err = clReleaseContext(context);
    if(err < 0) { printError(rank, "Couldn't release context"); }
}

/**
 * Function to release OpenCL object memory to avoid memory leaks.
 * @param rank: Rank of the current process. Used for error messaging.
 * @param low: The starting index of the portion to be sorted.
 * @param high: The ending index of the portion to be sorted.
 */
void copy_kernel_args(int rank, int low, int high)
{
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufA);
    if (err < 0) { printError(rank, "Couldn't set kernel argument bufA"); }
    clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&low);
    if (err < 0) { printError(rank, "Couldn't set kernel argument size"); }
    clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&high);
    if (err < 0) { printError(rank, "Couldn't set kernel argument startRow"); }
}

/**
 * Function to create OpenCL buffers for kernel computation.
 * @param rank: Rank of the current process. Used for error messaging.
 * @param m1: 1st Vector to copy to kernel.
 * @param m2: 2nd Vector to copy to kernel.
 */
void setup_kernel_memory(int rank, std::vector<int> m1, std::vector<int> m2)
{
    bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, m1.size() * sizeof(int), NULL, &err);
    if (err < 0) { printError(rank, "Couldn't create buffer A"); }
    bufB = clCreateBuffer(context, CL_MEM_READ_WRITE, m2.size() * sizeof(int), NULL, &err);
    if (err < 0) { printError(rank, "Couldn't create buffer A"); }
    // Copy matrices to the GPU
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, m1.size() * sizeof(int), m1.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) { printError(rank, "Couldn't write to buffer A"); }
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, m2.size() * sizeof(int), m2.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) { printError(rank, "Couldn't write to buffer A"); }
}

// Function provided previously to create device object for current platform.
cl_device_id create_device()
{
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    {
        perror("Couldn't identify a platform");
        exit(1);
    }

    // Access a device
    // GPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND)
    {
        // CPU
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err < 0)
    {
        perror("Couldn't access any devices");
        exit(1);
    }

    return dev;
}

//Provided function to build provided .cl program.
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file

    Creates a program from the source code in the filename file.
    Specifically, the code reads the file's content into a char array
    called program_buffer, and then calls clCreateProgramWithSource.
    */
    program = clCreateProgramWithSource(ctx, 1,
                                        (const char **)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program

    The fourth parameter accepts options that configure the compilation.
    These are similar to the flags used by gcc. For example, you can
    define a macro with the option -DMACRO=VALUE and turn off optimization
    with -cl-opt-disable.
    */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

// Function to setup OpenCL components.
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname) {
    device_id = create_device();
    cl_int err;
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    program = build_program(context, device_id, filename);

    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    };

    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0) {
        perror("Couldn't create a kernel");
        printf("error =%d", err);
        exit(1);
    };

}

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

    if(argc > 1)
    {
        size = std::stoi(argv[1]);
    }

    int worldRank, worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    std::vector<int> arr(size);

    if(worldRank == 0)
    {
        randomVector(arr, 0, 10);
    }

    global[1] = size;

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

    // Buffers for sending data to the device.
    std::vector<int> localArr(sendCounts[worldRank]), recvArr(sendCounts[worldRank]);
    if(worldRank == 0)
    {
        start = std::chrono::high_resolution_clock::now();
    }

    // Distribute the array among all processes using MPI_Scatterv
    MPI_Scatterv(arr.data(), sendCounts.data(), displs.data(), MPI_INT,
                 localArr.data(),sendCounts[worldRank], MPI_INT, 0, MPI_COMM_WORLD);


    setup_openCL_device_context_queue_kernel((char *) "./quickSort.cl", (char *) "quickSort");
    setup_kernel_memory(worldRank,  localArr, recvArr);
    copy_kernel_args(worldRank, 0, localArr.size());

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printError(worldRank, "Error enqueuing kernel");
    }
    clWaitForEvents(1, &event);
    err = clEnqueueReadBuffer(queue, bufB, CL_TRUE, 0, sendCounts[worldRank] * sizeof(int),
                              localArr.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printError(worldRank, "Error enqueuing buffer read");
    }

    // Ensure cl queue is completed before gathering data.
    clFinish(queue);

    // Gather the sorted subarrays from all processes back to the root process.
    MPI_Gatherv(localArr.data(), sendCounts[worldRank], MPI_INT, arr.data(),
                sendCounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    // If the current process is the root, merge the sorted subarrays and stop the clock.
    if(worldRank == 0)
    {
        arr = mergeKSortedArrays(arr, displs);

        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time Taken my OpenCL Quicksort: " << duration.count() << " microseconds" << std::endl;

        //std::cout << "Sorted Array: " << std::endl;
        //printVector(arr);

        bool sorted = isSorted(arr);
        if(!sorted)
        {
            std::cout << "Is Sorted: " << std::boolalpha << sorted << std::endl;
        }
        writeToCSV("output.csv", "OpenCL_Quicksort", size, duration.count(), sorted);
    }

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;

}