def dynamic_sliding_window(arr, x: int) ->int :
    min_length = float('inf')
    
    #The current range and sum of our sliding window
    start = 0
    end = 0
    current_sum = 0
    while end < len(arr):
        current_sum = current_sum + arr[end]
        end = end + 1
        
        while start < end and current_sum >= x:
            current_sum = current_sum - arr[start]
            start = start + 1
            
            min_length = min(min_length, end-start+1)
    return min_length
arr = [1,4,2,3,5]
x = 5
print(dynamic_sliding_window(arr, x))
    