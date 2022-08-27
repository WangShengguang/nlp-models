# 0.5/1000
# 市场营销团队，
#
# 双塔
#
#
#
# // 评测题目: 无


#   xxxxxx
#   xxxoxx
#   xxoxox
#   xxxoxx
#   xxxxxx

#   output = 5


#   xxoxox
#   xxxoxx
#   xxxxxx
#   xxxxxx
#   xxxxxx

#   output=3

#   oooxxxxxxxxx
#   oxxxxxxxxxxx
#   oxxxxxxxxxxx
#   xxxxxxxxxxxx
#   xxxxxxxxxxxx


def dfs(i, j, matrix, visited):
    if (i, j) not in visited:
        visited.add((i, j))
        for i in [i + 1, i - 1]:
            for j in [j - 1, j + 1]:
                if 0 <= i <= len(matrix) and 0 <= j <= len(matrix[0]) and matrix[i][j] != '棋子':
                    dfs(i, j, matrix, visited)


def get_aera(matrix):
    all_fail_visited = set()
    num_cols, num_rows = len(matrix[0]), len(matrix)
    for i in range(num_rows):
        for j in range(num_cols):
            if i == 0 or j + 1 == num_rows and matrix[i][j] != '棋子':  # 边界点，从边界点开始去标记所有点
                if (i, j) not in all_fail_visited:
                    visited = set()
                    dfs(i, j, matrix, visited)
                    all_fail_visited.update(visited)
    #
    count = num_cols * num_rows - len(all_fail_visited)
    return count

# def get_aera(matrix):

#   top = left = 0
#   down = len(matrix)
#   right = len(matrix[0])
#   i=j=0
#   # 最外层初始化
#   for j in range(right):
#     if matrix[i][j]!='棋子':
#     	matrix[i][j]=0
#     top+=1
#     for i in range(down):
#       if matrix[i][j]!='棋子':
#     	matrix[i][j]=0
#     right-=1
#     for j in range(right, -1, -1):
#       if matrix[i][j]!='棋子':
#     	matrix[i][j]=0
#     down-=1
#     for i in range(down, -1, -1):
#       if matrix[i][j]!='棋子':
#     	matrix[i][j]=0
#     left+=1
#     # 内层遍历
#     while left<=right:
#       for j in range(right):
#       if matrix[i][j]!='棋子':
#         if matrix[i-1][j]==0 or matrix[i][j-1]==0 or matrix[i+1][j]==0 or matrix[i][j+]==0:
#           matrix[i][j]=0
#       top+=1
#       for i in range(down):
#         if matrix[i][j]!='棋子':
#           if matrix[i-1][j]==0 or matrix[i][j-1]==0 or matrix[i+1][j]==0 or matrix[i][j+]==0:
#           	matrix[i][j]=0
#       right-=1
#       for j in range(right, -1, -1):
#         if matrix[i][j]!='棋子':
#           if matrix[i-1][j]==0 or matrix[i][j-1]==0 or matrix[i+1][j]==0 or matrix[i][j+]==0:
#           	matrix[i][j]=0
#       down-=1
#       for i in range(down, -1, -1):
#         if matrix[i][j]!='棋子':
#           if matrix[i-1][j]==0 or matrix[i][j-1]==0 or matrix[i+1][j]==0 or matrix[i][j+]==0:
#          	matrix[i][j]=0
#       left+=1
# 	#
#     #标记结束
#     #统计剩下的点
#     count = 0
#     for i in range(len(matrix)):
#       for j in range(len(matrix[0])):
#         if matrix[i][j]=='棋子' or matrix[i][j]!=0:
#           count+=1
#     return count

# coding=utf-8
import sys

# str = input()
# print(str)
print('Hello,World!')


class Solution(object):
    def __init__(self, matrix):
        self.matrix = matrix
        self._prepare()

    def _prepare(self):
        num_row, num_col = len(self.matrix), len(self.matrix[0])
        for i in range(1, num_row):
            self.matrix[i][0] += self.matrix[i - 1][0]
        for j in range(1, num_col):
            self.matrix[0][j] += self.matrix[0][j - 1]
        for i in range(1, num_row):
            for j in range(1, num_col):
                self.matrix[i][j] += self.matrix[i][j - 1] + self.matrix[i - 1][j] - self.matrix[i - 1][j - 1]

    def caculate(self, top_left, down_right):
        top, left = top_left
        #         top-=1
        #         left-=1
        down, right = down_right
        #         down-=1
        #         right-=1
        #
        down_left_val = top_right_val = top_left_val = 0
        if left > 0:
            down_left_val = self.matrix[down][left - 1]
        if top > 0:
            top_right_val = self.matrix[top - 1][right]
        if top > 0 and left > 0:
            top_left_val = self.matrix[top - 1][left - 1]
        #         print(self.matrix[down][right]==45, down_left_val==12 , top_right_val==0 , top_left_val==0 )
        return (self.matrix[down][right] -
                down_left_val - top_right_val + top_left_val)


def main():
    matrix = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]
    obj = Solution(matrix)
    print(obj.matrix)
    res = obj.caculate((0, 1), (2, 2))
    print(res, res == 33)


main()




#

import numpy as np

def eighty_to_ninty(img):
    """
    把图像中80像素的位置转成90像素
    """
    img=img.astype(np.int32)
    img[img!=2]=0
    return img
"""
zijier
脉脉
直播策略算法：属于春招补招；

---------
畅游无限：有nlp岗位

阿里安全？20210705-15:00： 
9个球，找出最重；两次：11

阿里安全：推荐；内容识别，违规过滤；赌博；电商，ugc，评论，文娱等；阿里健康；
文本过滤等，问诊对话；语气态度等；
搜广推；ai治理（过滤垃圾风险数据）；query脱钩等；内容理解相关；tf；



"""