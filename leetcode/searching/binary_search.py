import math

def binarySearch(nums, x):
    low = 0
    high = len(nums) - 1
    while low <= high:
        mid = int(low + (high - low) / 2)
        if(x == nums[mid] ): 
            return nums.index(x)
        elif( x > nums[mid]):
            low = mid + 1
        elif( x < nums[mid]):
            high = mid - 1
    return -1
    # return nums

nums = [-1,0,3,5,9,12]
x = 12
low = 0
high = len(nums) - 1
print("before calling: ", nums, x, low, high)
print("after calling: ",binarySearch( nums, x));    

# 1- first + last /2
# 2- add while until low <= high
# 3- initialize mid = low + (high - low) / 2
# 4- if element == arr(mid) return mid
# 5- if search element > arr(mid), low = mid + 1
# 6- if search element > arr(mid), high = mid - 1 