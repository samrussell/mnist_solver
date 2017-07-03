import base_trainer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.utils import to_categorical
from keras.datasets import cifar10

class CifarCnnTrainer(base_trainer.BaseTrainer):
  cifar_data = None
  num_classes = 10
  img_rows = 32
  img_cols = 32
  img_channels = 3
  epochs = 12

  def build_model(self, input_shape):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
    model.add(Conv2D(96, (3, 3), activation='relu'))
    model.add(Conv2D(96, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Conv2D(192, (3, 3), activation='relu'))
    model.add(Conv2D(192, (3, 3), activation='relu'))
    model.add(Conv2D(192, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Conv2D(192, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    self.model = model

  def test_results(self, testing_values, testing_labels):
    score = self.model.evaluate(testing_values, testing_labels, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

  def load_training_data(self):
    self.load_cifar_data()
    (x_train, y_train), (x_test, y_test) = self.load_cifar_data()
    
    shaped_labels = to_categorical(y_train, self.num_classes)
    scaled_values = self.scale_values(x_train)
    shaped_values = self.reshape_values(scaled_values)

    return shaped_values, shaped_labels

  def load_testing_data(self):
    (x_train, y_train), (x_test, y_test) = self.load_cifar_data()
    
    shaped_labels = to_categorical(y_test, self.num_classes)
    scaled_values = self.scale_values(x_test)
    shaped_values = self.reshape_values(scaled_values)

    return shaped_values, shaped_labels

  def load_cifar_data(self):
    if not self.cifar_data:
      self.cifar_data = cifar10.load_data()

    return self.cifar_data

if __name__ == "__main__":
  CifarCnnTrainer().run()