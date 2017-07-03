import base_trainer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras

class BiggerCnnTrainer(base_trainer.BaseTrainer):
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

    return model

if __name__ == "__main__":
  BiggerCnnTrainer().run()