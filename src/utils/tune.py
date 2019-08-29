import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

input_shape = (28,28,1)
num_classes = 10
images_directory = '../../data/sudoku_set'

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.load_weights('../models/model_fnt_2.hd5')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0003),
              metrics=['accuracy'])

datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_generator = datagen.flow_from_directory(
                            images_directory,
                            target_size=(28,28),
                            color_mode='grayscale',
                            batch_size=32,
                            class_mode='categorical',
                            subset='training',
                            shuffle=True)

validation_generator = datagen.flow_from_directory(
                                images_directory,
                                target_size=(28,28),
                                color_mode='grayscale',
                                batch_size=32,
                                class_mode='categorical',
                                subset='validation',
                                shuffle=True)

checkpoint_name = '../models/model_fnt_3.hd5'
checkpoint  = keras.callbacks.ModelCheckpoint(checkpoint_name, 
                                              monitor='val_acc',
                                              save_best_only=True,
                                              verbose=1)


model.fit_generator(train_generator,
                    epochs=15,
                    validation_data=validation_generator,
                    callbacks=[checkpoint])

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, 225 // 32+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = [str(i) for i in range(10)]
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))