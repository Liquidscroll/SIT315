#!/bin/bash

mpicxx -std=c++17 ./MPI_Quicksort.cpp -o ./MPI_Quicksort
mpicxx -std=c++17 ./OpenCL_Quicksort.cpp -lOpenCL -o ./OpenCL_Quicksort