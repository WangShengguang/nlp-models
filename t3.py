"""
https://www.showmebug.com/pads/GBATMK 
"""


class Node:
    def __init__(self, value=None):
        self.value = value
        self.next: Node = None


def reversed_link(link: Node):
    p = link
    q = link.next
    while q:
        r = q.next
        q.next = p
        q = p
        p = r


def main():
    pass


if __name__ == '__main__':
    main()
