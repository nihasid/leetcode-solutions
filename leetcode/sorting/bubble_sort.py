def bubbleSort(arr):
    n = len(arr)
    
    for i in range(n):
        for j in range(0, n-i-1):
            if(arr[j] > arr[j+1]):
                arr[j], arr[j+1] = arr[j+1], arr[j]
                
    return arr
    
arr = [23, 1, 2, 5,22, 14, 9, 6, 4]
print(bubbleSort(arr))