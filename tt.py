class Node(object):
    def __init__(self, var):
        self.var = var
        self.left: Node = None
        self.right: Node = None


def order(node: Node, pre_seq=[], mid_seq=[]):
    if node is not None:
        pre_seq.append(node)
        order(node.left)
        mid_seq.append(node)
        order(node.right)


def is_same(t1: Node, t2: Node):
    pre_seq1, mid_seq1 = [], []
    order(t1, pre_seq1, mid_seq1)
    #
    pre_seq2, mid_seq2 = [], []
    order(t2, pre_seq2, mid_seq2)
    return pre_seq1 == pre_seq2 and mid_seq1 == mid_seq2


def func(scores):
    # if len(scores) <= 1:
    #     return len(scores)
    # elif len(scores) == 2:
    #     if scores[1] != scores[0]:
    #         return 2
    #     return 1
    #
    dp = [1 for _ in range(len(scores))]
    for i in range(1, len(scores) - 1):
        if scores[i - 1] < scores[i]:
            dp[i] = dp[i - 1] + 1
        elif scores[i - 1] > scores[i]:
            dp[i - 1] += 1
            # 继续调整
    return sum(dp)


def main():
    pass


if __name__ == '__main__':
    main()
