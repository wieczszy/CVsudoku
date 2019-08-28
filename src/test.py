import numpy as np
from classfier import Classifier
from reader import ImageReader
from solver import SudokuSolver


if __name__ == '__main__':

    file_path = "../data/test/sudoku005.png"
    reader = ImageReader()
    cells = reader.extract_board_cells(file_path)
    cells = cells.reshape(-1,28, 28, 1)

    classifier = Classifier('model.h5')
    classifications = classifier.classify_cells(cells)
    classifications = [str(c) for c in classifications]
    grid = ''.join(classifications)

    solver = SudokuSolver()
    solver.solve(grid)
