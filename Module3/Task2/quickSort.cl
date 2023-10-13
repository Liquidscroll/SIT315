// Swap function
void swap(__global int* a, __global int* b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Median of Three function
int medianOfThree(__global int* arr, int low, int high)
{
    int mid = (low + high) / 2;

    if(arr[low] > arr[high]) {
        swap(&arr[low], &arr[high]);
    }
    if(arr[low] > arr[mid]) {
        swap(&arr[low], &arr[mid]);
    }
    if(arr[mid] < arr[high]) {
        swap(&arr[mid], &arr[high]);
    }
    return arr[high];
}

// Partition function
int partition(__global int* arr, int low, int high)
{
    int pivot = medianOfThree(arr, low, high);
    int i = (low - 1);

    for(int j = low; j <= high - 1; j++)
    {
        if(arr[j] <= pivot)
        {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

__kernel void quickSort(__global int* arr, int low, int high)
{

    const int STACK_SIZE = 100000;
    int stack[STACK_SIZE]; // Cannot set dynamically, so we will make this a large number.
    int stackIndex = 0;

    // Push initial low and high onto stack
    stack[stackIndex] = low;
    stackIndex++;
    stack[stackIndex] = high;
    stackIndex++;
    while(stackIndex > 0)
    {
        // Pop high and low off stack
        stackIndex--;
        high = stack[stackIndex];
        stackIndex--;
        low = stack[stackIndex];

        int pivot = partition(arr, low, high);

        if(pivot - 1 > low)
        {
            stack[stackIndex] = low;
            stackIndex++;
            stack[stackIndex] = pivot - 1;
            stackIndex++;
        }

        if(pivot + 1 < high)
        {
            stack[stackIndex] = pivot + 1;
            stackIndex++;
            stack[stackIndex] = high;
            stackIndex++;
        }
    }
}
