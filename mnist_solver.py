'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
# from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K
import pandas
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='training data CSV', required=True)
parser.add_argument('--test', help='test data CSV', required=True)
parser.add_argument('--output', help='output data CSV', required=True)
commandline_args = parser.parse_args()

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
training_dataframe = pandas.read_csv(commandline_args.train)
values = training_dataframe.values[:,1:]
labels = to_categorical(training_dataframe.values[:,0], num_classes)
values = values.astype('float32')
values /= 255

validation_dataframe = pandas.read_csv(commandline_args.test)
validation_values = validation_dataframe.values.astype('float32')
validation_values /= 255

if K.image_data_format() == 'channels_first':
    # this might learn them sideways... not that it matters
    values = values.reshape(values.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    values = values.reshape(values.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# keep 1% as validation
validation_split_index = int(values.shape[0] * 0.99)

train_values = values[:validation_split_index,:]
test_values = values[validation_split_index:,:]
train_labels = labels[:validation_split_index,:]
test_labels = labels[validation_split_index:,:]

print('values shape:', values.shape)
print(train_values.shape[0], 'train samples')
print(test_values.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# training
model.fit(train_values, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_values, test_labels))
# testing
predictions = model.predict(validation_values)
df = pandas.DataFrame(data=np.argmax(predictions, axis=1), columns=['Label'])
df.insert(0, 'ImageId', range(1, 1 + len(df)))

# save results
df.to_csv(commandline_args.output, index=False)
