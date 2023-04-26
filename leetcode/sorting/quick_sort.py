def quick_sort(arr):
    import sys
    print(arr)
    # Base case: return the array if it's empty or contains only one element
    if len(arr) <= 1:
        return arr
    print(':::::::Quick Sort called::::::::::::')
    # Choose a pivot element (usually the last element in the array)
    pivot = arr[-1]
    print('pivot', pivot)
    # Initialize left and right pointers
    left = []
    right = []

    # Partition the array into two parts based on the pivot element
    for i in range(len(arr)-1):
        print('********* array[i]: ', arr[i], 'i: ', i, '********************')
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
        print('left: ', left)
        print('right: ', right) 
    # # Recursively sort the left and right sub-arrays
    print('***************left::::*****', pivot)
    sorted_left = quick_sort(left)
    sorted_right = quick_sort(right)
    
    # print('sorted_left: ', sorted_left, 'sorted_right: ', sorted_right, 'pivot: ', [pivot])
    # # Concatenate the sorted left sub-array, pivot, and sorted right sub-array
    return sorted_left + [pivot] + sorted_right

# Example usage:
arr = [19, 17, 4, 11, 10, 13]

sorted_arr = quick_sort(arr)
print(sorted_arr)
