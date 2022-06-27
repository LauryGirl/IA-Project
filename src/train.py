import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers, backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D 

K.clear_session()

data_train = '../dataset/training'
data_validation = '../dataset/validation'

#params
epochs = 20
height, length = 100, 100
batch_size = 32
steps = 1000
steps_validation = 200
filters_conv1, filters_conv2, filters_conv3, filters_conv4, filters_conv5, filters_conv6 = 32, 64, 128, 256, 512, 1024
length_filter1 = (3,3) 
length_filter2 = (2,2)
length_pool = (2,2)
class_ = 3
lr = 0.0005

#preprocessing images
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.3, zoom_range = 0.3, horizontal_flip = True)
validation_datagen = ImageDataGenerator(rescale=1./255)

training_images = train_datagen.flow_from_directory(data_train, target_size = (height, length), batch_size = batch_size, class_mode = 'categorical')
validation_images = validation_datagen.flow_from_directory(data_validation, target_size = (height, length), batch_size = batch_size, class_mode = 'categorical')

# CNN create
cnn = Sequential()

cnn.add(Convolution2D(filters_conv1, length_filter1, padding = 'same', input_shape = (height, length, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=length_pool))

cnn.add(Convolution2D(filters_conv2, length_filter2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=length_pool))

cnn.add(Convolution2D(filters_conv3, length_filter2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=length_pool))

cnn.add(Convolution2D(filters_conv4, length_filter2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=length_pool))

cnn.add(Convolution2D(filters_conv5, length_filter2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=length_pool))

cnn.add(Convolution2D(filters_conv6, length_filter2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=length_pool))

#fully connected layers
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(class_, activation='softmax'))


cnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

cnn.fit(training_images, steps_per_epoch=steps, epochs=epochs, validation_data=validation_images, validation_steps= steps_validation)

dir = '../model'
if not os.path.exists(dir):
    os.mkdir(dir)

cnn.save('../model/model.h5')
cnn.save_weights('../model/weigth.h5')