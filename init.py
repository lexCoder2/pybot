import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from model import model_LSM

from strategies.strategies import Strategies 
from tensorflow.keras.callbacks import ModelCheckpoint

MODEL_NAME = 'train/cp.ckpt'
csv_file = './bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv'
PERIODS_FORWARD = 60
DATA_SIZE = 3_000_000
def main():
  # get data
  df_ = Frame(csv_file) 
  df= df_.get_df()

  # clean and normalize data
  s = Strategies(df)
  l = s.calc_all()

  df['bollingerU'] = l[0]['BOLU']
  df['bollingerD'] = l[0]['BOLD']
  df['ema'] = l[1]
  df['osc_esotcK'] = l[2]['K']
  df['osc_esotcD'] = l[2]['D']
  df['rsi'] = l[3]
  df['sma'] = l[4]
  is_plot = False
  df.drop('Timestamp', axis=1, inplace=True)

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
  



  # test model



  




  if is_plot:  
    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
    plt.plot(range(df.shape[0]),df['Volume_(Currency)'])
    # plt.plot(range(df.shape[0]), df['max'])
    # plt.plot(range(df.shape[0]), df['min'])

    # plt.plot(range(df.shape[0]), calc_bollinger['BOLU'])
    # plt.plot(range(df.shape[0]), calc_bollinger['BOLD'])
    # plt.plot(range(df.shape[0]), calc_sma)
    # plt.plot(range(df.shape[0]), calc_ema)

    # plt.plot(range(df.shape[0]), calc_osc_esotc['K'])
    # plt.plot(range(df.shape[0]), calc_osc_esotc['D'])
    
    # plt.plot(range(df.shape[0]), calc_rsi)
    
    plt.xticks(range(0,df.shape[0],int(DATA_SIZE /100_000)),df['Timestamp'].loc[::int(DATA_SIZE /100_000)],rotation=45)
    plt.xlabel('Timestamp',fontsize=12)
    plt.ylabel('Mid Price',fontsize=12)
    plt.show()
  

  # plt.plot(l[0].range,s)
 

def percentage_analisis(df):
  indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=PERIODS_FORWARD)
  df['max'] = df.rolling(window=indexer, min_periods=1)['High'].max()
  df['max'] = ((df['max'] - df['Close'])*100 ) / df['Close']
  print(df['max'][lambda x: x > 5].shape)

  df['min'] = df.rolling(window=indexer, min_periods=1)['Low'].min()
  df['min'] = ((df['min'] - df['Close'])*100 ) / df['Close']
  print(df['min'][lambda x: x > -1.5].shape)

def plot(df):
  plt.figure(figsize = (18,9))
  plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
  # plt.plot(range(df.shape[0]), df['max'])
  # plt.plot(range(df.shape[0]), df['min'])
  plt.xticks(range(0,df.shape[0],5000),df['Timestamp'].loc[::5000],rotation=45)
  plt.xlabel('Timestamp',fontsize=12)
  plt.ylabel('Mid Price',fontsize=12)
  plt.show()

class Frame:

  def __init__(self, csv) -> None:
    self.csv_file = csv
    self.get_data()
    self.clean_data()

  def get_data(self):
    self.df = pd.read_csv(self.csv_file).tail(DATA_SIZE)

  def get_df(self):
    return self.df

  def clean_data(self):
    self.df= self.df.interpolate()
    self.df.dropna(how="all", subset=['Open' ,'High','Low','Close','Volume_(BTC)'], inplace=True)
    self.df['Timestamp']= pd.to_datetime(self.df['Timestamp'], unit='s' )
    

def MA(df, periods):
  return df.rolling(window=periods).mean()

def EMA(df, periods):
  df['4dayEWM'] = df['sales'].ewm(span=periods, adjust=False).mean()

def PM(n):
  pass

if __name__ == "__main__":
  main()