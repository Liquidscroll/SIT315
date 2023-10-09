mpicxx ./MPI_ParallelMultiplication.cpp -o MPI_Multi
mpicxx -fopenmp ./OMP_MPI_ParallelMultiplication.cpp -o OMP_MPI_Multi
mpicxx -pthread ./OpenCL_MPI_ParallelMultiplication.cpp -lOpenCL -o OpenCL_MPI_Multi