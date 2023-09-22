#include <omp.h>

#include "ParallelQuickSort.h"

namespace ParallelQuickSort {

    void swap(int &a, int &b)
    {
        int temp = a;
        a = b;
        b = temp;
    }
    /**
     * Finds and returns the median of three values in an array.
     * @param arr[] The array containing the integers.
     * @param low The index of the first integer.
     * @param high The index of the second integer.
     */
    int medianOfThree(int arr[], int low, int high)
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
     * Partitions the array around a pivot and places the elements smaller
     * than the pivot on the left, and larger on the right.
     * @param arr[] The array to be partitioned.
     * @param low The starting index of the portion to be partitioned.
     * @param high The ending index of the portion to be partitioned.
     */
    int partition(int arr[], int low, int high)
    {
        int pivot = medianOfThree(arr, low, high);
        int i = (low - 1);
        for(int j = low; j <= high - 1; j++)
        {
            if (arr[j] < pivot)
            {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        return (i + 1);
    }
    /**
     * Sorts the array using the quicksort algorithm in parallel.
     * @param arr[] The array to be sorted.
     * @param low The starting index of the sorting range.
     * @param high The ending index of the sorting range.
     */
    void quickSort(int arr[], int low, int high)
    {
        while(low < high) {
            int part = partition(arr, low, high);

            int lowPart = part - 1;
            int highPart = part + 1;
            //Sort smaller array first
            if(part - low < high - part)
            {
                // We create a task for the call to quickSort for
                // the smaller sub-array, spawning off threads for them.
                // The main thread continues with the larger sub-array
                // and partitions it.
#pragma omp task default(none) shared(arr) firstprivate(low, lowPart)
                quickSort(arr, low, lowPart);
                low = highPart;
            } else {
#pragma omp task default(none) shared(arr) firstprivate(high, highPart)
                quickSort(arr, highPart, high);
                high = lowPart;
            }
        }
    }
}