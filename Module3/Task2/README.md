### Matrix Multiplication

Build using the commands:

```
MPI Only:
mpicxx -std=c++17 ./MPI_Quicksort.cpp -o ./MPI_Quicksort

MPI + OpenCL:
mpicxx -std=c++17 ./OpenCL_Quicksort.cpp -lOpenCL -o ./OpenCL_Quicksort
```

Or through the bash script provided:

```
./build.sh
```

Run using the commands:

```
MPI Only:
mpiexec -np 2 -hostfile ./cluster ./MPI_Quicksort

MPI + OpenCL:
mpiexec -np 2 -hostfile ./cluster ./OpenCL_Quicksort
```

Or through the bash script provided:

```
./run.sh
```