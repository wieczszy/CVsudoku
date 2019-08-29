import cv2
import os
import random
from classfier import Classifier
from reader import ImageReader


def save_cells(file_path, model_path='models/model_fnt_2.hd5'):
    reader = ImageReader()
    cells = reader.extract_board_cells(file_path)
    classifier = Classifier(model_path)
    blanks = classifier._idenitfy_blanks(cells)
    
    for i in range(len(cells)):
        if not blanks[i]:
            fname = '../data/windows/window_{}.png'.format(str(random.random()).split('.')[1])
            cv2.imwrite(fname, cells[i])


def extract():
    test_dir = '../data/test2/'
    files = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]

    for f in files:
        try:
            save_cells(f)
        except:
            pass


if __name__ == '__main__':
    model_path = 'models/model_fnt_2.hd5'
    c = Classifier(model_path)
    
