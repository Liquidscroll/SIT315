#!/bin/bash

mpiexec -np 2 -hostfile ./cluster ./MPI_Quicksort 
mpiexec -np 2 -hostfile ./cluster ./OpenCL_Quicksort 