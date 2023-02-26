# Initialize this with the 9x9 array of digits
# None element in the 9x9 array represents empty cell
# solve() solves the sudoku and saves it in self.digits
# The solved sudoku can be accessed by the digits member of this class
from string import digits
from z3 import *

class Z3Solver:
    def __init__(self, digits):
        self.digits = digits

    # Solve function
    # Returns True if sudoku admits a solution
    # False otherwise
    # Solved sudoku can be found in self.digits
    def solve(self):

        s = Solver()

        grid =[
                [
                Int(f"cell_{r}_{c}") for c in range(9)
                ]
                    for r in range(9)
            ]   

        for r in range(9):
            for c in range(9):
                s.add(grid[r][c] >= 1, grid[r][c] <= 9)
            s.add(Distinct(grid[r]))

        for c in range(9):
            s.add(Distinct([grid[r][c] for r in range(9)]))

        for x in range(3):
            for y in range(3):
                s.add(Distinct(
                    [
                        grid[x*3][y*3],
                        grid[x*3][y*3+1],
                        grid[x*3][y*3+2],
                        grid[x*3+1][y*3],
                        grid[x*3+1][y*3+1],
                        grid[x*3+1][y*3+2],
                        grid[x*3+2][y*3],
                        grid[x*3+2][y*3+1],
                        grid[x*3+2][y*3+2],

                    ]
                ))
        for r in range(9):
            for c in range(9):
                if self.digits[r][c] != 0:
                    s.add(grid[r][c] == self.digits[r][c])
        status = s.check()
        m = s.model()
        if status == sat:
            for r in range(9):
                for c in range(9):
                    if self.digits[r][c] == 0:
                        self.digits[r][c] = m.eval(grid[r][c])
            return True
        else: return False
        

    
        
