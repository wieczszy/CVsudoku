import cv2
import numpy as np
from classfier import Classifier
from reader import ImageReader
from solver import SudokuSolver


def test_solving(file_path, model_path):
    reader = ImageReader()
    cells = reader.extract_board_cells(file_path)
    
    classifier = Classifier(model_path)
    classifications = classifier.classify_cells(cells)
    classifications = [str(c) for c in classifications]
    grid = ''.join(classifications)

    solver = SudokuSolver()
    solver.solve(grid)

def test_gird_localization(file_path):
    reader = ImageReader()
    board = reader.extract_board_cells(file_path, test=True)
    cv2.imwrite('../data/board_test.png', board)

def test_digit_recognition(file_path, model_path):
    reader = ImageReader()
    cells = reader.extract_board_cells(file_path)
    print(len(cells))

    classifier = Classifier(model_path))
    classifications = np.array(classifier.classify_cells(cells)).reshape((9,9))
    print('\n')
    print(classifications)

def test_cells(file_path):
    reader = ImageReader()
    cells = reader.extract_board_cells(file_path)

    for i in range(len(cells)):
        cv2.imwrite('../data/windows/cell_{}.jpg'.format(i+1), cells[i])


if __name__ == '__main__':
    test_file = '../data/test/sudoku001.jpg'
    model_path = 'models/model_fnt_2.hd5'

    # test_solving(test_file, model_path)
    # test_gird_localization(test_file)
    # test_digit_recognition(test_file, model_path)
    # test_cells(test_file)