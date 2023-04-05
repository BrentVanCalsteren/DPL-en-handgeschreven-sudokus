from Sudoku import *
from Checker import *

import random


class RandomGenerator:

    @staticmethod
    def generateRandomTileRecursive(sudoku, col, row):
        valids = set(i + 1 for i in range(sudoku.dim ** 2))

        while len(valids):
            newItem = random.choice(list(valids))
            valids.remove(newItem)
            sudoku.setTile(col, row, Tile(newItem))

            if Checker.checkSudoku(sudoku):
                if row + 1 >= sudoku.dim ** 2:
                    newRow = 0
                    newCol = col + 1
                else:
                    newRow = row + 1
                    newCol = col

                if newCol >= sudoku.dim ** 2:
                    return True

                if RandomGenerator.generateRandomTileRecursive(sudoku, newCol, newRow):
                    return True

        sudoku.setTile(col, row, Tile())
        return False

    @staticmethod
    def generateRandomSudoku(dim=9):
        sudoku = Sudoku(dim)
        RandomGenerator.generateRandomTileRecursive(sudoku, 0, 0)
        sudoku.setTiles()
        return sudoku
