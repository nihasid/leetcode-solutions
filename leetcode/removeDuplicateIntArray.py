def removeDuplicate(nums):
    left = 1
    for right in range(1, len(nums)):
        if(nums[right] != nums[right - 1]):
            nums[left] = nums[right]
            left += 1
        
    return left
        
    
array = [1,1,2]
print(removeDuplicate(array))