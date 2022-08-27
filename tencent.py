"""
两个数字的字符串，计算相加之后得到的字符串，当数很大的时候用这种方法计算高效。例如，string a=“453”，string b=“29”，add（a，b）输出“482”

"""
"""
两个数字的字符串，计算相加之后得到的字符串，
当数很大的时候用这种方法计算高效。
例如，string a=“453”，string b=“29”，add（a，b）输出“482”

"""


def add(a, b):
    short, long = a, b
    if len(a) > len(b):
        short, long = b, a
    # print(len(short), len(long))
    results = []
    left = 0
    for i in range(-1, -len(short) - 1, -1):
        t = int(short[i]) + int(long[i]) + left
        left, n = divmod(t, 10)
        results.append(n)
        # print(i)
    for i in range(-len(short) - 1, -len(long) - 1, -1):
        # print(i)
        t = int(long[i]) + left
        left, n = divmod(t, 10)
        results.append(n)
    #
    return ''.join([str(n) for n in results[::-1]])


def main():
    a = "453"
    b = "29"
    res = add(a, b)
    print(res)
    # 输出“482”


if __name__ == '__main__':
    main()

"""
互动，游戏数据挖掘；游戏社交网络、图谱构建、社交分析、活跃预测； 
离线：tensorflow  
上线：C++，go；spark，hive sql； 
------------ ------------
------------ ------------


"""
