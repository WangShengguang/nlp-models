
"""
1.     编程：

给定一个N （1<N<20)，打印一个NxN的矩阵，值从1到NxN；打印的最终结果从左上角1开始，沿着外围不停加1，直到最里面。譬如3, 则打印

1     8     7
2     9     6
3     4     5

譬如4，则打印：

1  12   11   10
2  13   16    9
3  14   15    8
4  5    6    7
程序编译调试通过，需要发源代码（不限编程语言）回来（最好是文件）。

"""


def print_matrix(n:int):
    left=top = 0
    right = down = n-1
    num = 1
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    i=j=0
    while left<=right:
        for i in range(top, down+1):
            matrix[i][j] = num
            num+=1
        left+=1
        for j in range(left, right+1):
            matrix[i][j] = num
            num+=1
        down-=1
        for i in range(down, top-1, -1):
            matrix[i][j] = num
            num+=1
        right-=1
        # print('num: ',i,j, num)
        # print(matrix)
        for j in range(right, left-1, -1):
            matrix[i][j] = num
            num+=1
        top+=1
    #
    for i in range(n):
        for j in range(n):
            print(matrix[i][j], end=' ')
        print()

def main():
    # print_matrix(3)
    print_matrix(4)


if __name__ == '__main__':
    main()

"""
夸克搜索：搜索部门；北京；负责相关性；算法组做推荐；小说推荐；
上线cpp，简单代码；  Python  


relu 零太多 

1---5；

1---5；

1---5；

111 -》 1 
123 -》 8 



为什么这么多零没有乘法变成零。

加法是concat的特殊情况；

-----------
新浪新闻的，新闻推荐NLP组，特征提取、推荐、排序等
一个月： 

---
背包问题


"""