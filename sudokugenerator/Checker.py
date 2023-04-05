class Checker:
    @staticmethod
    def checkSudoku(sudoku, failInfo=None):
        if failInfo is None:
            failInfo = {}
        for i in range(sudoku.dim ** 2):
            if not Checker.check(sudoku.getRow(i), sudoku.dim ** 2):
                failInfo["row"] = i
                return False
            if not Checker.check(sudoku.getColumn(i), sudoku.dim ** 2):
                failInfo["col"] = i
                return False
        for i in range(sudoku.dim):
            for j in range(sudoku.dim):
                if not Checker.check(sudoku.matrixToList(sudoku.getBlock(i, j)), sudoku.dim ** 2):
                    failInfo["block"] = [i, j]
                    return False
        return True

    @staticmethod
    def check(row, length):
        checkList = [0]*length
        for element in row:
            if not element.isEmpty():
                checkList[element.value - 1] += 1
        return max(checkList) <= 1

    @staticmethod
    def isSolved(sudoku):
        for i in range(sudoku.dim ** 2):
            for number in range(sudoku.dim ** 2):
                if not (number + 1) in sudoku.getColumnValues(i):
                    # print(" nummer " + str(number + 1) +
                    #      " komt niet voor in kolom " + str(i) + " de sudoku is niet opgelost")
                    return False
                if not (number + 1) in sudoku.getRowValues(i):
                    # print(" nummer " + str(number + 1) +
                    #      " komt niet voor in rij " + str(i) + " de sudoku is niet opgelost")
                    return False
        for i in range(sudoku.dim):
            for j in range(sudoku.dim):
                for number in range(sudoku.dim ** 2):
                    if not (number + 1) in sudoku.getBlockValues(i, j):
                        # print(" nummer " + str(number + 1) +
                        #      " komt niet voor in block " + str(i) + " , " + str(j) + " de sudoku is niet opgelost")
                        return False
        return True
