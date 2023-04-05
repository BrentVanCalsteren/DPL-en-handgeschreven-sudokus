import math


def find_empty(sudoku):
    size = len(sudoku)
    for i in range(size):
        for j in range(size):
            if sudoku[i][j] == 0:
                return (i,j)
    return None


def valid(sudoku, pos, num):
    size = len(sudoku)
    sq = int(math.sqrt(size))
    for i in range(size):
        if sudoku[i][pos[1]] == num:
            return False

    for j in range(size):
        if sudoku[pos[0]][j] == num:
            return False

    start_i = pos[0] - pos[0] % sq
    start_j = pos[1] - pos[1] % sq
    for i in range(sq):
        for j in range(sq):
            if sudoku[start_i + i][start_j + j] == num:
                return False
    return True

def solve(sudoku):
    # True if valid suk
    size = len(sudoku)
    empty = find_empty(sudoku)
    if not empty:
        return True

    for nums in range(1, size+1):
        if valid(sudoku, empty, nums):
            sudoku[empty[0]][empty[1]] = nums

            if solve(sudoku):
                return True
            sudoku[empty[0]][empty[1]] = 0  # wrong -> 0
    return False
