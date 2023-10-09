
echo "mpiexec -np 2 -hostfile ./cluster ./MPI_Multi $1"
mpiexec -np 2 -hostfile ./cluster ./MPI_Multi "$1"
echo "mpiexec -np 2 -hostfile ./cluster ./OMP_MPI_Multi $1"
mpiexec -np 2 -hostfile ./cluster ./OMP_MPI_Multi "$1"
echo "mpiexec -np 2 -hostfile ./cluster ./OpenCL_MPI_Multi $1"
mpiexec -np 2 -hostfile ./cluster ./OpenCL_MPI_Multi "$1"


