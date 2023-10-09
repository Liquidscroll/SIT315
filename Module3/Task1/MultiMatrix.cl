/**
 * @param   maxRow  The maximum number of rows in the input matrices.
 * @param   maxCol  The maximum number of columns in the input matrices.
 * @param   v1      Pointer to the first input matrix, represented as a 1D array.
 * @param   v2      Pointer to the second input matrix, represented as a 1D array.
 * @param   v3      Pointer to the output matrix, where the result will be stored, represented as a 1D array.
 */
__kernel void matrixMul(const int maxRow, const int maxCol,
                      const __global int* v1,const __global int* v2,__global int* v3) {

    // Get the global thread identifiers:
    // 'i' corresponds to the row index and 'j' corresponds to the column index in the resultant matrix.
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    // Perform boundary check to ensure the thread (i, j) is within the dimensions of the resultant matrix.
    // If the thread indices are out of bound, exit the kernel for this particular thread.
    if(i > maxRow - 1 || j > maxCol - 1) { return; }

    // Declare and initialize a variable 'res' to accumulate the sum of products for the resultant matrix element.
    int res = 0;

    // Iterate over the 'k' index, summing up the product of the corresponding elements from matrices v1 and v2.
    // v1[i * maxCol + k]: accesses the element in the i-th row and k-th column of matrix v1.
    // v2[k * maxCol + j]: accesses the element in the k-th row and j-th column of matrix v2.
    for(int k = 0; k < maxCol; k++)
    {
        res += v1[i * maxCol + k] * v2[k * maxCol + j];
    }

    //uncomment to see the index each PE works on
    //printf("Kernel process index :(%d,%d)\n1d index in C: %d\nres: %d\n", i, j, i * maxCol + j, res);
    //printf("res: %d\n", res);
    v3[i * maxCol + j] = res;
}