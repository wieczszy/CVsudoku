import cv2
import numpy as np
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
    cells = reader.extract_board_cells(file_path)

    classifier = Classifier(model_path)
    classifications = classifier.classify_cells(cells)
    classifications = [str(c) for c in classifications]
    grid = ''.join(classifications)

    solver = SudokuSolver()
    solver.solve(grid)