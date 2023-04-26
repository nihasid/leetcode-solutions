
def missingNumber( nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    i = 0
    j = 1

    while True:
        if i in nums and j not in nums:
            return j
        elif i not in nums and j in nums:
            return i
        i += 1
        j += 1
    # import numpy as np
    # sorted_num = np.sort(nums)
    # a = int()
    # for i in range(0, np.max(sorted_num)+1):
    #     if sorted_num[i] != i:
    #         a = i 
    #         break
    #     else: a = i+1
    # # if a == 'null': a = np.max(sorted_num) + 1
    # return a  

nums = [0,1,2]
print(missingNumber( nums))