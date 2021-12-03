import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler
from historical_data import HistoricalData
from model import model_LSM
from plot import graph, plot_strategies
from strategies.strategies import Strategies 
from tensorflow.keras.callbacks import ModelCheckpoint

def time(years = 0, months = 0, days=0, hours = 0, minutes = 0):
  return ( (years * 1440 * 365) + (months * 1440 * 30) + (days * 1440) + (hours *  60) + (minutes) )

MODEL_NAME = 'train/cp.ckpt'
csv_file = './archive(1)/xrpusd.csv'
PERIODS_FORWARD = 60
DATA_SIZE = time(
  years = 0,
  months = 10,
  days = 0
  )


def main():
  # get data
  df_ = HistoricalData(csv_file, limit_rows=DATA_SIZE) 
  df= df_.df
  
  # calculate strategies
  s = Strategies(df)
  df = df.join(pd.DataFrame(s.calc_all('4hour')))
  
  df_day = df_.get_data_day()

  if False:
    s_day = Strategies(df_day)
    df_day = df_day.join(pd.DataFrame(s_day.calc_all('days')))
    ema55day = df_day.supports.resample('1min').bfill()
 
  wide = 240
  df = windowed_predictions(df, wide)

  
  # plot data
  plot_strategies(df)
  graph(df, bar=False)
  
  return
  
  #  divide train and test data
  train_data = df[:math.ceil(df.shape[0] *.8)]
  test_data = df[math.ceil(df.shape[0] *.8):]
  train_data = train_data.values

  # shape data 
  scaler = MinMaxScaler()
  smoothing_window_size = math.ceil(DATA_SIZE / 5)
 
  for di in range(0,DATA_SIZE - smoothing_window_size*2, smoothing_window_size):
      scaler.fit(train_data[di:di+smoothing_window_size,:])
      train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

  # You normalize the last bit of remaining data
  scaler.fit(train_data[di+smoothing_window_size:,:])
  train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

  # create the enter values
  history_length = 60
  x_train = []
  y_train = []
  per = 0
  for i in range (history_length, len(train_data) - 6):
    
    max_high = np.max(train_data[i:i + 6][1])
    min_high = np.min(train_data[i:i + 6][1])
    close_price = train_data[i][3]
    if close_price != 0 and min_high != np.NaN:
      
    
        
      y_train.append( [
        max_high/close_price,
        min_high/close_price
      ])
      x_train.append(train_data[i-history_length: i])
      if i > per * (len(train_data)/100):
        per = per + 1

        print('[' + str(per) + '%] [' + '='*per + '>' + ' '* (100 - per) + ']', end='\r')
  print ('[Complete 100%] [' + '=' * 100 + ']')


  x_train, y_train = np.array(x_train), np.array(y_train)
  print(x_train.shape)
  
  
  # create model
  model = model_LSM(x_train)

  # train model
  cp = ModelCheckpoint(MODEL_NAME, save_weights_only=True, verbose=1)
  trained = model.fit(x_train, y_train, batch_size=2, epochs=5, callbacks=[cp])
  



def windowed_predictions(df, wide):
  indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=wide)
  mhw = df.loc[:,'high'].rolling(window=indexer, min_periods=1).max()
  mlw = df.loc[:,'low'].rolling(window=indexer, min_periods=1).min()
  df.loc[:,'max_high_wide'] =  mhw
  df.loc[:,'min_low_wide'] = mlw
  return df

if __name__ == "__main__":
  main()