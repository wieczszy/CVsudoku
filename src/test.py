import numpy as np
from reader import ImageReader
from classfier import Classifier


if __name__ == '__main__':

    file_path = "../data/test/sudoku005.png"
    reader = ImageReader()
    cells = reader.extract_board_cells(file_path)
    cells = cells.reshape(-1,28, 28, 1)

    c = Classifier('model.h5')
    classifications = np.array(c.classify_cells(cells)).reshape((9,9))
    print(classifications)