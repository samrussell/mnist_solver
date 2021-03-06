import pandas
import argparse
import numpy as np
import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import mnist

class BaseTrainer:
  batch_size = 128
  num_classes = 10
  epochs = 12
  validation_percentage = 0.99
  img_rows = 28
  img_cols = 28
  img_channels = 1

  def __init__(self):
    # set up commandline arguments
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('--train', help='training data CSV', required=True)
    self.parser.add_argument('--test', help='test data CSV', required=True)
    self.parser.add_argument('--output', help='output data CSV', required=True)

  def run(self):
    self.load_args()
    shaped_values, shaped_labels = self.load_training_data()
    testing_values, testing_labels = self.load_testing_data()
    training_values, validation_values = self.split_data(shaped_values)
    training_labels, validation_labels = self.split_data(shaped_labels)

    print('values shape:', shaped_values.shape)
    print(training_values.shape[0], 'training samples')
    print(validation_values.shape[0], 'validation samples')

    self.build_model(input_shape=training_values.shape[1:])

    # training
    self.model.fit(training_values, training_labels,
              batch_size=self.batch_size,
              epochs=self.epochs,
              verbose=1,
              validation_data=(validation_values, validation_labels))

    self.test_results(testing_values, testing_labels)

  def test_results(self, testing_values, testing_labels):
    predictions = self.model.predict(testing_values)
    df = pandas.DataFrame(data=np.argmax(predictions, axis=1), columns=['Label'])
    df.insert(0, 'ImageId', range(1, 1 + len(df)))

    # save results
    df.to_csv(self.commandline_args.output, index=False)

  def load_args(self):
    self.commandline_args = self.parser.parse_args()

  def load_training_data(self):
    training_dataframe = pandas.read_csv(self.commandline_args.train)
    values = training_dataframe.values[:,1:]
    labels = training_dataframe.values[:,0]
    
    shaped_labels = to_categorical(labels, self.num_classes)
    scaled_values = self.scale_values(values)
    shaped_values = self.reshape_values(scaled_values)

    return shaped_values, shaped_labels

  def load_testing_data(self):
    testing_dataframe = pandas.read_csv(self.commandline_args.test)
    values = testing_dataframe.values
    
    scaled_values = self.scale_values(values)
    shaped_values = self.reshape_values(scaled_values)

    return shaped_values, None

  def scale_values(self, values):
    return values.astype('float32') / 255

  def reshape_values(self, values):
    # TODO make it work when data comes pre-shaped
    if K.image_data_format() == 'channels_first':
        reshaped_values = values.reshape(values.shape[0], self.img_channels, self.img_rows, self.img_cols)
    else:
        reshaped_values = values.reshape(values.shape[0], self.img_rows, self.img_cols, self.img_channels)

    return reshaped_values

  def split_data(self, data):
    landmark = int(data.shape[0] * self.validation_percentage)
    return data[:landmark], data[landmark:]

  def build_model(self, input_shape):
    raise NotImplementedError("Must be implemented by subclass")
