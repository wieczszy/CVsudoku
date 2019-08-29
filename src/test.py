import numpy as np
from classfier import Classifier
from reader import ImageReader
from solver import SudokuSolver


def test_solving(file_path):
    reader = ImageReader()
    cells = reader.extract_board_cells(file_path)
    
    classifier = Classifier('model.h5')
    classifications = classifier.classify_cells(cells)
    classifications = [str(c) for c in classifications]
    grid = ''.join(classifications)

    solver = SudokuSolver()
    solver.solve(grid)


if __name__ == '__main__':
    test_solving('../data/test/sudoku005.png')