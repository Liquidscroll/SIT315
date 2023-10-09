### Matrix Multiplication

Build using the commands:

```
MPI Only:
mpicxx ./MPI_ParallelMultiplication.cpp -o MPI_Multi

MPI + OMP:
mpicxx -fopenmp ./OMP_MPI_ParallelMultiplication.cpp -o OMP_MPI_Multi

MPI + OpenCL:
mpicxx -pthread ./OpenCL_MPI_ParallelMultiplication.cpp -lOpenCL -o OpenCL_MPI_Multi
```

Or through the bash script provided:

```
./build.sh
```

Run using the commands:

```
MPI Only:
mpiexec -np 2 -hostfile ./cluster ./MPI_Multi "$1"

MPI + OMP:
mpiexec -np 2 -hostfile ./cluster ./OMP_MPI_Multi "$1"

MPI + OpenCL:
mpiexec -np 2 -hostfile ./cluster ./OpenCL_MPI_Multi "$1"
```

Or through the bash script provided:

```
./run.sh
```