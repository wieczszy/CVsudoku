import cv2
import numpy as np
import logging
from argparse import ArgumentParser
from src.classifier import Classifier
from src.reader import ImageReader
from src.solver import SudokuSolver


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image", help="Input image")
    parser.add_argument("-w", "--weights", help="CNN weights", default="src/models/model.h5")
    args = parser.parse_args()

    file_path = args.image
    model_path = args.weights

    reader = ImageReader()
    
    try:
        cells = reader.extract_board_cells(file_path)
    except AttributeError:
        print()
        logging.error('\nThe image has not been read correctly - file not found!\n')
        exit(0)    

    try:
        classifier = Classifier(model_path)
        classifications = classifier.classify_cells(cells)
        classifications = [str(c) for c in classifications]
        grid = ''.join(classifications)
    except OSError:
        logging.error('\nThe model weights have not been loaded - file not found!\n')
        exit(0)

    solver = SudokuSolver()

    try:
        solver.solve(grid)
    except TypeError:
        logging.error('The image has not been read correctly - solution not found!\n')
        exit(0)