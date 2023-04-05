class Tile:
    value = 0
    pencilMarks = set()
    column = 0
    row = 0

    def __init__(self, value=0, pencilmarks=None, column=0, row=0):
        self.value = value
        if not pencilmarks:
            self.pencilMarks = set()
        else:
            self.pencilMarks = pencilmarks
        self.column = column
        self.row = row

    def copy(self):
        return Tile(self.value, self.pencilMarks, self.column, self.row)

    def getRow(self):
        return self.row

    def getColumn(self):
        return self.column

    def coors(self):
        return[self.column, self.row]


    def setValue(self, value):
        self.value = value
        self.pencilMarks = set()

    def isEmpty(self):
        return self.value == 0

    def __eq__(self, other):
        return self.value == other.value and self.pencilMarks == other.pencilMarks \
               and self.column == other.column and self.row == other.column

    def __hash__(self):
        return self.column + self.row * 9
