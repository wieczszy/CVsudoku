import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import cv2
import numpy as np


class Classifier():
    def __init__(self, weights):
        self.model_def = weights

    def _get_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=(28,28,1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

        model.load_weights(self.model_def)
        return model

    def _idenitfy_blanks(self, cells):
        blanks = []
        for cell in cells:
            num_white_px = np.sum(cell == 255)
            if num_white_px == 0:
                blanks.append(True)
            else:
                blanks.append(False)
        return blanks

    def classify_cells(self, cells):
        cells = cells.reshape(-1,28, 28, 1)
        model = self._get_model()
        classifications = []
        blanks = self._idenitfy_blanks(cells)
        predictions = model.predict(cells)
        predictions = [np.argmax(p) for p in predictions]

        for i in range(len(predictions)):
            if not blanks[i]:
                classifications.append(predictions[i])
            else:
                classifications.append(0)
        return classifications
        