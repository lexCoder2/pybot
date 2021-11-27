from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.recurrent import LSTM


def model_LSM(x_train):

  model = Sequential()

  model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(2))
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model