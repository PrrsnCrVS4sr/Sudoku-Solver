# Initialize this with the 9x9 array of digits
# None element in the 9x9 array represents empty cell
# solve() solves the sudoku and saves it in self.digits
# The solved sudoku can be accessed by the digits member of this class
from string import digits


class Solver:
    def __init__(self, digits):
        self.digits = digits

    # Solve function
    # Returns True if sudoku admits a solution
    # False otherwise
    # Solved sudoku can be found in self.digits
    def solve(self):
        pos = self.find_empty()
        if not pos:
            return True
        else:
            x,y = pos
            for i in range(1, 10):
                if(self.check_valid(pos, i)):
                    self.digits[x][y] = i
                    if(self.solve()):
                        return True
                    self.digits[x][y] = 0
        return False

    def find_empty(self):
        for i in range(9):
            for j in range(9):
                if self.digits[i][j] == 0:
                    return (i, j)
        return None

    def check_valid(self, pos, num):

        for i in range(9):
            if self.digits[i][pos[1]] == num and i != pos[0]:
                return False
        for i in range(9):
            if self.digits[pos[0]][i] == num and i != pos[1]:
                return False

        x = pos[1] // 3
        y = pos[0] // 3

        for i in range(y*3, y*3+3):
            for j in range(x*3, x*3 + 3):
                if self.digits[i][j] == num and i != pos[0] and j != pos[1]:
                    return False
        return True
    
        
