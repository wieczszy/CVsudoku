import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

input_shape = (28,28,1)
num_classes = 10
images_directory = '../../data/new_fnt'

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

model.load_weights('../models/model.h5')

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


checkpoint_name = '../models/model_fnt_2.hd5'
checkpoint  = keras.callbacks.ModelCheckpoint(checkpoint_name, 
                                              monitor='val_acc',
                                              save_best_only=True,
                                              verbose=1)

model.fit_generator(train_generator,
                    epochs=10,
                    validation_data=validation_generator,
                    callbacks=[checkpoint])
