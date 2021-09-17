import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from our_functions import *

#--- data upload
train_images, train_labels, test_images,test_labels = load_hack_data()
#train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)

#--- thresholding
Threshold = 25
train_images = (train_images > Threshold)
test_images = (test_images > Threshold)

num_filters = 64
kernel_size = 4
strides = 1
pool_size = 2
nodes = 1024
image_shape = train_images.shape[1:]

model = Sequential()
#-----------------First----------------------------------
model.add(Conv2D(32, kernel_size, strides, padding="valid",input_shape = image_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size, padding='valid'))
#-----------------Second---------------------------------
model.add(Conv2D(num_filters, kernel_size, strides, padding="valid"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size, padding='valid'))
#-----------------Third----------------------------------
model.add(Flatten())
model.add(Dense(nodes))
model.add(Activation("relu"))
#------------output--------------------------------------
model.add(Dense(2))
model.add(Activation('sigmoid'))
#--------------------------------------------------------
#opt = SGD(learning_rate=0.0001, momentum=0.9)
#model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics= ['accuracy'])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
#model.summary()

#--------- model training -------------
history = model.fit(train_images, train_labels,batch_size=128, epochs=20, validation_split = 0.010)

#--------- model predicts test data labels ------
predict_labels = model.predict(test_images)
generate_sample_file('result.json',predict_labels)

