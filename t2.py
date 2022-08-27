def _get_area(i, j, num_row, num_col, matrix):
    """广度遍历;  """
    area = 0
    visited = set()
    if matrix[i][j] == 0:
        return area, visited
    area += 1
    queue = [(i, j)]
    visited.add((i, j))
    while i < num_row and j < num_col and queue:
        i, j = queue.pop(0)
        for i, j in [(i + 1, j), (i + 1, j + 1), (i - 1, j), (i - 1, j - 1),
                     (i, j - 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]:
            if matrix[i][j] == 1 and not (i, j) in visited:
                area += 1
                queue.append((i, j))
                visited.add((i, j))
    return area, visited


def get_max_area(matrix):
    if not matrix or not matrix[0]:
        return 0
    num_row, num_col = len(matrix), len(matrix[0])
    max_area = 0
    all_visited = set()
    for i in range(num_row):
        for j in range(num_col):
            if (i, j) not in all_visited:
                area, visited = _get_area(i, j, num_row, num_col, matrix)
                max_area = max(max_area, area)
                all_visited.update(visited)
    return max_area


def main():
    pass


if __name__ == '__main__':
    main()
