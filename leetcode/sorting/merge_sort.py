def merge_sort(arr, level = 0):
    if len(arr) <= 1:
        return arr
    # print('#######################calling#######################' * level)
    # Split the array in half
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Recursively sort the left and right halves
    left_half = merge_sort(left_half, level+1)
    right_half = merge_sort(right_half, level+1)

    # Merge the sorted left and right halves
    merged = []
    left_idx, right_idx = 0, 0
    while left_idx < len(left_half) and right_idx < len(right_half):
        if left_half[left_idx] < right_half[right_idx]:
            merged.append(left_half[left_idx])
            left_idx += 1
        else:
            merged.append(right_half[right_idx])
            right_idx += 1
    # print(level * ' - LEVEL - ', left_half[:left_idx])
    # Add any remaining elements from the left and right halves
    merged.extend(left_half[left_idx:])
    # print('*********left extended **********')
    
    # print(level * ' - LEVEL - ', right_half[right_idx])
    merged.extend(right_half[right_idx:])
    # print('*********right extended **********')

    return merged

# Example usage:
arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
sorted_arr = merge_sort(arr)
print(sorted_arr)
