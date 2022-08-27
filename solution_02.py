"""
2. 算法描述：

有n个正整数（每个数长度可能不一样，但都小于99999999），将它们表示成字符串形式。
对每一个字符串s，可以翻转为新字符串s'，如“1234”可以翻转成“4321”。
现在，将这n个字符串以任意顺序拼接成一个长整数，每个字符串可以选择是否翻转。
请问，如何才能得到最大的长整数？

用文字描述下解决这个问题的过程、步骤等。
"""


# 过程
# 先预处理，保留 s和s' 中对应数字较大的那一个
# 比较每个字符串的第一位，大的拼接到新字符串结尾；若有几个字符串当前位置相等，这几个字符串先进行比较处理逻辑，处理完后再处理剩下的字符串；
# 对这几个第一位相等的字符串，再比较第二位，根据第二位是否存在，大，小，相等来进行进一步比较处理
# 具体处理步骤如下

# 算法描述：
# 步骤 1.对输入字符串做预处理：  将每个字符串s与反转后的s'对应的数字对比，保留对应数字更大的那一个
# 步骤 2. 初始化新数字为空字符串；当前位置设置为 0
# 步骤 3. 若当前列表非空，比较所有传入字符串在当前位置对应的数字
# 步骤 4. case1：若存在最大数字；将当前位置数字最大字符串的拼接到新数字末尾；并移除比较列表，当前位置+1，回到步骤 3
# 步骤 5. case2: 若其中有几个字符串当前位置对应的数字相同；
#           case2.1: 若存在下一位；继续比较这几个字符串的下一位，将第下一位数字较大的字符串拼接到新数字末尾；
#                     并从比较列表移除，当前位置+1，比较当前剩下的这几个字符串中当前位置对应的数字，回到 步骤 3
#           case2.2: 若其中一位数字已经没有下一位；将其拼接到新数字末尾；并从比较列表移除，
#                   当前位置+1，比较当前剩下的这几个字符串中当前位置对应的数字，回到 步骤 3
# 步骤 6. 若待处理字符串字符串为空，结束，否则，返回步骤 3
from collections import defaultdict


def solution02(num_strs):
    num_strs = [s if int(s)>int(s[::-1]) else s[::-1]
                for s in num_strs]
    lens = {len(s) for s in num_strs}
    max_len, min_len = max(lens), min(lens)
    # lens = sorted(lens)
    # max_len, min_len  = lens[-1], lens[0]
    dp = defaultdict(list) # dp[i] 长度为 i 的列表的排序
    for _len in lens:
        for num in num_strs:
            if len(num)>=_len:
                dp[_len].append(num[:_len])

    #
    for num in num_strs:
        dp[len(num)].append(num)


    # def position_wise_compare(nums):
    #     for n in nums:
    #
    #
    # position = 0
    # while num_strs:
    #     li = defaultdict(list)
    #     for i in range(len(num_strs)):
    #
    #         if num_strs[i][]:
    #
    #     for i in range(max_len):
    #         if num_strs[i]
"""

6位数；个位和40；被8 
999940 
推荐模型； 
auc指标不变；提高泛化； 

夸克搜索；推荐业务，小说推荐，教育推荐；搜索rank阶段；

[][]

----------------
百度搜索策略部 
北京：
---
数学、算法、理论、编程项目等；


"""