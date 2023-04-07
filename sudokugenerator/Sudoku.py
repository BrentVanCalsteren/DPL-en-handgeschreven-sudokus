import math
import random

from Tile import Tile


class Sudoku:
    dim = 3
    matrix = [[]]

    def __init__(self, dim=9):
        self.dim = int(math.sqrt(dim))
        self.matrix = [[Tile() for _i in range(self.dim ** 2)] for _j in range(self.dim ** 2)]
        self.setTiles()

    def getTile(self, column, row):
        return self.matrix[column][row]

    def setTile(self, column, row, tile):
        self.matrix[column][row] = tile
        self.matrix[column][row].column = column
        self.matrix[column][row].row = row

    def __eq__(self, other):
        for col in range(self.dim):
            for row in range(self.dim):
                if not self.matrix[col][row] == other.matrix[col][row]:
                    return False
        return True

    @staticmethod
    def isTileEmpty(tile):
        return tile.isEmpty()

    def getColumn(self, column):
        return self.matrix[column]

    def getRegions(self):
        regions = []

        for i in range(self.dim ** 2):
            regions.append(self.getColumn(i))
            regions.append(self.getRow(i))
        for x in range(self.dim):
            for y in range(self.dim):
                regions.append(self.getBlockAsList(x, y))

        return regions

    def getColumnSet(self, column):
        return set(self.matrix[column])

    def getColumnValues(self, column):
        return [i.value for i in self.matrix[column]]

    def getColumnValuesSet(self, column):
        return {i.value for i in self.matrix[column]}

    def getColumnEmptys(self, column):
        return [i for i in self.matrix[column] if i.isEmpty()]

    def getColumnPencils(self, column):
        return {1, 2, 3, 4, 5, 6, 7, 8, 9} - self.getColumnValuesSet(column)

    @staticmethod
    def getListValues(l):
        return [s.value for s in l]

    @staticmethod
    def getMatrixValues(m):
        return [[x.value for x in row] for row in m]

    @staticmethod
    def emptyOrNot(x):
        i = random.randint(0, 1)
        if i:
            return x
        else:
            x.value = 0
            return x

    @staticmethod
    def generateRandomEmptys(sudoku):
        return [[Sudoku.emptyOrNot(x) for x in row] for row in sudoku.matrix]

    @staticmethod
    def invalidOrNot(x, size):
        i = random.randint(0, 1)
        j = random.randint(1, size)
        if i or x.value == 0:
            return x
        else:
            x.value = j
            return x

    @staticmethod
    def makeRandomInvalid(sudoku, size):
        return [[Sudoku.invalidOrNot(x, size) for x in row] for row in sudoku.matrix]

    @staticmethod
    def getSetValues(s):
        return {t.value for t in s}

    @staticmethod
    def getSetPencils(s):
        return {1, 2, 3, 4, 5, 6, 7, 8, 9} - Sudoku.getSetValues(s)

    @staticmethod
    def getEmptyInRegion(region):
        return [v for v in region if v.isEmpty()]

    def getRow(self, row):
        return [i[row] for i in self.matrix]

    def getRowValues(self, row):
        return [i[row].value for i in self.matrix]

    def getRowValuesSet(self, row):
        return {i[row].value for i in self.matrix}

    def getRowEmptys(self, row):
        return [i[row] for i in self.matrix if i[row].isEmpty()]

    def getRowPencils(self, column):
        return {1, 2, 3, 4, 5, 6, 7, 8, 9} - self.getRowValuesSet(column)

    def getRowSet(self, row):
        return {i[row] for i in self.matrix}

    def getBlock(self, col, row):
        return [[self.matrix[col * self.dim + i][row * self.dim + j] for j in
                 range(self.dim)] for i in range(self.dim)]

    def getBlockAsList(self, col, row):
        return [self.matrix[col * self.dim + i][row * self.dim + j] for i in
                range(self.dim) for j in range(self.dim)]

    def getBlockAsSet(self, col, row):
        return set(self.getBlockAsList(col, row))

    def getBlockValues(self, col, row):
        return [self.matrix[col * self.dim + i][row * self.dim + j].value for i in
                range(self.dim) for j in range(self.dim)]

    def getBlockEmptys(self, col, row):
        return [self.matrix[col * self.dim + i][row * self.dim + j] for i in
                range(self.dim) for j in range(self.dim) if
                self.matrix[col * self.dim + i][row * self.dim + j].isEmpty()]

    def getBlockValuesSet(self, col, row):
        return {self.matrix[col * self.dim + i][row * self.dim + j].value for i in
                range(self.dim) for j in range(self.dim)}


    @staticmethod
    def matrixToList(matrix):
        return [i for j in matrix for i in j]


    def printText(self):
        print(sudokugenerator.SudokuParser.toText(self))

    # sudoku manipulaties

    def permutateNumbers(self, perm):
        sudoku = Sudoku()
        for col in range(self.dim ** 2):
            for row in range(self.dim ** 2):
                if self.getTile(col, row).value:
                    sudoku.matrix[col][row] = Tile(perm[self.getTile(col, row).value - 1], set())
                else:
                    sudoku.matrix[col][row] = Tile(0, set())
        sudoku.setTiles()
        return sudoku

    def permutateCols(self, perm):
        sudoku = Sudoku()
        for col in range(self.dim ** 2):
            for row in range(self.dim ** 2):
                sudoku.matrix[col][row] = Tile(self.getTile(perm[col] - 1, row).value, set())

        sudoku.setTiles()
        return sudoku

    def permutateRows(self, perm):
        sudoku = Sudoku()
        for col in range(self.dim ** 2):
            for row in range(self.dim ** 2):
                sudoku.matrix[col][row] = Tile(self.getTile(col, perm[row] - 1).value, set())

        sudoku.setTiles()
        return sudoku

    def mirror(self):
        sudoku = Sudoku()
        for col in range(self.dim ** 2):
            for row in range(self.dim ** 2):
                sudoku.matrix[col][row] = Tile(self.getTile(row, col).value, set())

        sudoku.setTiles()
        return sudoku

    def copy(self):
        sudoku = Sudoku()
        sudoku.matrix = [[tile.copy() for tile in row] for row in self.matrix]

        return sudoku

    def setTiles(self):
        for i in range(self.dim ** 2):
            for j in range(self.dim ** 2):
                self.matrix[i][j].column = i
                self.matrix[i][j].row = j
