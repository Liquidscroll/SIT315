

#ifndef PARALLEL_QUICKSORT_H
#define PARALLEL_QUICKSORT_H

namespace ParallelQuickSort
{
    void swap(int &a, int &b);
    int medianOfThree(int arr[], int low, int high);
    int partition(int arr[], int low, int high);
    void quickSort(int arr[], int low, int high);
}

#endif
