from .Sudoku import *


class SudokuParser:

    @staticmethod
    def parseFromFile(path, separator=" "):
        file = open(path, "r")
        sudoku = SudokuParser.parse("".join(file.readlines()), separator)
        file.close()
        return sudoku

    @staticmethod
    def parse(string, separator):
        string = string.replace("\n", separator)

        line = string.split(separator)
        line = list(filter(None, line))  # Filter empty strings
        line = list(map(lambda x: int(x), line))
        sudoku = Sudoku()
        sudoku.matrix = SudokuParser.listToMatrix(line)
        sudoku.setTiles()

        return sudoku

    @staticmethod
    def listToMatrix(l):
        matrix = []
        matrixDim = math.sqrt(len(l))
        currentCol = []

        for i in range(len(l)):
            if i % matrixDim == 0:
                currentCol = []
                matrix.append(currentCol)
            if l[i] == 0:
                pencilMarks = set()
                currentCol.append(Tile(0, pencilMarks))
            else:
                currentCol.append(Tile(l[i], set()))

        return matrix

    @staticmethod
    def toText(sudoku):
        s = ""
        for i in range(sudoku.dim ** 2):
            for tile in sudoku.getColumn(i):
                if tile.value:
                    s += str(tile.value) + " "
                else:
                    s += "0 "
            s += "\n"
        return s
