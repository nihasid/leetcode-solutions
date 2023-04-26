
def single_num( nums):
    counts = {}
    for i in nums:
        if nums.count(i) == 1:
            return i
        
    return -1
    # for num in nums:
    #     print("nums array: ", nums)
    #     if num in counts:
    #         counts[num] += 1
    #         print("if count: ",counts)
    #     else:
    #         counts[num] = 1
    #         print("else count: ",counts)
        
            
    # for num in nums:
    #     if counts[num] == 1:
    #         return num
    # return -1
    

nums = [2,2,1,4,6,2,3,5,7]
print(single_num( nums))

