
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <vector>
#include <thread>
#include <random>
#include <mpi.h>
#include <CL/cl.h>

//using namespace std;

int SZ = 4;
const int TS = 4;

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

size_t local[2];
size_t global[2]; // number of rows and cols or basically the number of threads with indices i and j where i is the row and j is the col of the matric C

cl_device_id create_device();
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);

// Forward declare OpenCL Methods.
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);
void setup_kernel_memory(int rank, std::vector<int> m1, std::vector<int> m2, std::vector<int> m3);
void copy_kernel_args(int rows, int rank);
void free_memory(int rank);

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
/**
 * Initialize a 1D vector for matrix of given rows and cols.
 * @param rows: of the matrix.
 * @param cols: of the matrix.
 */
std::vector<int> initArray(int rows, int cols)
{
    std::vector<int> arr(rows * cols);
    return arr;
}
/**
 * Initialize the given matrix with random values.
 * @param matrix: The matrix to be initialized.
 * @param rows: of the matrix.
 * @param cols: of the matrix.
 * @param low: Lower bound for random values.
 * @param high: Upper bound for random values.
 */
void randomMatrix(std::vector<int> &matrix, int rows, int cols, int low, int high) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(low, high);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = dist(rng);
        }
    }
}
/**
 * Print the given matrix to the console.
 * @param matrix: The matrix to be printed.
 * @param rows: of the matrix.
 * @param cols: of the matrix.
 */
void printMatrix(std::vector<int> &matrix, int rows, int cols)
{
    std::string res(cols * 2, '=');
    res += "\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res += std::to_string(matrix[i * cols + j]) + " ";
        }
        res += "\n";
    }
    std::string foot(cols * 2, '=');
    res += foot;
    printf("%s\n", res.c_str());
}

void printError(int worldRank, std::string message)
{
    printf("%s\n", message.c_str());
    printf("Rank: %d -- Error: %d\n", worldRank, err);
    exit(1);
}
int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    if (argc > 1)
        SZ = atoi(argv[1]);

    // Setting global and local work sizes.
    global[0] = SZ;
    global[1] = SZ;
    local[0] = TS;
    local[1] = TS;

    //printf("Size: %d\n", SZ);

    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::vector<int> sendCounts(worldSize), displs(worldSize);
    std::vector<int> v1, v2, v3;

    // Generate data on root process.
    if (worldRank == 0) {
        v1 = initArray(SZ, SZ);
        v2 = initArray(SZ, SZ);
        v3 = initArray(SZ, SZ);
        randomMatrix(v1, SZ, SZ, 0, 10);
        randomMatrix(v2, SZ, SZ, 0, 10);
        //printMatrix(v1, SZ, SZ);
        //printMatrix(v2, SZ, SZ);
    } else {
        v2.resize(SZ * SZ);
        v3.resize(SZ * SZ);
    }

    int rowsPerProcess = SZ / worldSize;
    int rem = SZ % worldSize;
    int sum = 0;

    int startRow = worldRank * rowsPerProcess;
    int endRow = (worldRank * rowsPerProcess) + rowsPerProcess;

    // Calculate the number of elements to be sent to each process and the associated displacement within the array.
    // If there is a remainder when dividing size by processes, then we can distribute an additional row to each
    // process, who's rank is below the remainder.
    for (int i = 0; i < worldSize; i++) {
        endRow += i < rem ? 1 : 0;
        sendCounts[i] = (rowsPerProcess + (i < rem ? 1 : 0)) * SZ;
        displs[i] = sum * SZ;
        sum += rowsPerProcess + (i < rem ? 1 : 0);
    }

    // Each process will receive a number of elements in v1 and the same number to v3 (the results matrix.)
    // So we create sub-arrays to hold these elements.
    std::vector<int> v1_sub(sendCounts[worldRank]),
            v3_sub(sendCounts[worldRank]);

    if (worldRank == 0) { start = std::chrono::high_resolution_clock::now(); }

    // Scatter the data to each process. We need to give MPI a pointer to beginning of the data buffers, and we do
    // this by using .data(). We use MPI_Scatterv as each process will receive a different number of elements.
    MPI_Scatterv(v1.data(), sendCounts.data(), displs.data(), MPI_INT,
                 v1_sub.data(), sendCounts[worldRank], MPI_INT, 0, MPI_COMM_WORLD);
    // v2 will be used by all processes, so we need to broadcast it.
    MPI_Bcast(v2.data(), SZ * SZ, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Setup the matrix OpenCL program and the kernel for each process.
    setup_openCL_device_context_queue_kernel((char *) "./MultiMatrix.cl", (char *) "matrixMul");
    setup_kernel_memory(worldRank,  v1_sub, v2, v3_sub);
    copy_kernel_args(sendCounts[worldRank] / SZ, worldRank);

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printError(worldRank, "Error enqueuing kernel");
    }
    clWaitForEvents(1, &event);
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sendCounts[worldRank] * sizeof(int),
                        v3_sub.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printError(worldRank, "Error enqueuing kernel");
    }

    // Ensure cl queue is completed before gather data.
    clFinish(queue);
    MPI_Gatherv(v3_sub.data(), sendCounts[worldRank], MPI_INT, v3.data(),
               sendCounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        printf("OpenCL Took: %lld microseconds\n",
               static_cast<long long int>(duration.count()));
        //printMatrix(v3, SZ, SZ);
        bool sorted = true;
        for(int i = 0; i < SZ; i++)
        {
            for(int j = 0; j < SZ; j++)
            {
                int res = 0;
                for(int k = 0; k < SZ; k++)
                {
                    res += v1[i * SZ + k] * v2[k * SZ + j];

                }
                if(v3[i * SZ + j] != res)
                {
                    printf("Error: %d != %d\n", v3[i * SZ + j], res);
                    printf("At Position: (%d, %d)\n", i, j);
                    sorted = false;
                }
            }
        }
        writeToCSV("output.csv", "OpenCL", SZ, duration.count(), sorted ? "true" : "false");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    free_memory(worldRank);

    MPI_Finalize();
    return 0;
}

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

void copy_kernel_args(int rows, int rank)
{
    //NOTE that we modified the first parameter (Rows of A)
    clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&rows);
    if (err < 0) { printError(rank, "Couldn't set kernel argument size"); }
    clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&SZ);
    if (err < 0) { printError(rank, "Couldn't set kernel argument startRow"); }
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufA);
    if (err < 0) { printError(rank, "Couldn't set kernel argument bufA"); }
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufB);
    if (err < 0) { printError(rank, "Couldn't set kernel argument bufB"); }
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&bufC);
    if (err < 0) { printError(rank, "Couldn't set kernel argument bufC"); }

    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

void setup_kernel_memory(int rank, std::vector<int> m1, std::vector<int> m2, std::vector<int> m3)
{
    //NOTE that we modified the bufA to only cover rows of A and C
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, m1.size() * sizeof(int), NULL, &err);
    if (err < 0) { printError(rank, "Couldn't create buffer A"); }
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, m2.size()* sizeof(int), NULL, &err);
    if (err < 0) { printError(rank, "Couldn't create buffer B"); }
    bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, m3.size() * sizeof(int), NULL, &err);
    if (err < 0) { printError(rank, "Couldn't create buffer C"); }

    // Copy matrices to the GPU
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, m1.size() * sizeof(int), m1.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) { printError(rank, "Couldn't write to buffer A"); }

    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, m2.size() * sizeof(int), m2.data(), 0, NULL, NULL);
    if (err < 0) { printError(rank, "Couldn't write to buffer B"); }

    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, m3.size() * sizeof(int), m3.data(), 0, NULL, NULL);
    if (err < 0) { printError(rank, "Couldn't write to buffer C"); }
}

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

    Creates a program from the source code in the add_numbers.cl file.
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
