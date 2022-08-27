#   1/2 + 1/2 ** 3 +  #
#   1/2 **2 + 1/2 **4 + #
#   2:1 = '' #

def solution(nums, target):
    """
    有序数组；重复，小到大；一个数最后出现的位置
    """
    left = 0
    right = len(nums) - 1
    mid = -1
    while left <= right:
        mid = (left + right) // 2
        num = nums[mid]
        if target < num:
            right = mid - 1  # num在左侧
        elif target > num:  # 在右侧
            left = mid + 1
        else:
            break
    #
    for i in range(mid + 1, len(nums)):
        if nums[i] != target:
            return i - 1
    return mid


def bi_tree():
    """ 前序遍历 """
    pass
