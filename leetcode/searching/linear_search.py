def sentinelLinearSearch(array, key):
    last = array[len(array) - 1]
    array[len(array) - 1] = key
    counter_index = 0
    while array[counter_index] != key:
        counter_index += 1
    array[len(array) - 1] = last
    if counter_index < len(array) - 1 or last == key:
        return counter_index
    else:
        return -1
 
array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
key = -2
index = sentinelLinearSearch(array, key)
if index == -1:
    print(f"{key} is not found in the array: {array}")
else:
    print(f"{key} is found at index {index} in the array: {array}")
    

# Sentinel linear search Algorithm
# Set the last element of the array to the target value. This is known as the sentinel value.
# Set the index variable “i” to the first element of the array.
# Use a loop to iterate through the array, comparing each element with the target value.
# If the current element is equal to the target value, return the index of the current element.
# Increment the index variable “i” by 1 after each iteration of the loop.
# If the loop completes and the target value is not found, return -1 to indicate that the value is not present in the array.
