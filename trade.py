
import os
from binance.client import Client
from numpy.core.shape_base import block
import pandas as pd
import numpy as np
from time import sleep
import random
import asyncio
from binance import ThreadedWebsocketManager, AsyncClient, BinanceSocketManager, BinanceAPIException, BinanceOrderException
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler

from pandas.core.frame import DataFrame

from strategies.strategies import Strategies

api_key = os.environ.get('binance_api')
api_secret = os.environ.get('binance_secret')

# client = Client(api_key, api_secret)
# client.API_URL = 'https://testnet.binance.vision/api'
# data = pd.DataFrame(index=['time', 'Low', 'High', 'Open', 'Close', 'Volume', 'Trades'])



buys = []
total = 0

async def kline_listener(client):
  df = pd.DataFrame()
  bm = BinanceSocketManager(client)

  async with bm.kline_socket(symbol='BTCUSDT') as stream:
    while True:
      res = await stream.recv()
      is_final_candle = res['k']['x']
      if df.shape[0] > 30:
        trade(df.append(res))
      if is_final_candle:
        df = append_data_from_stream(df, res)
      



def trade(tail_market: DataFrame):
  margin_up = 1.05
  margin_down = .97
  last_price = tail_market.last
  for bought in buys:

    if (should_sell(bought, tail_market)):
      sell_market(bought)
    else:
      bought.minuteinfo.append(last_price)
  if should_buy(tail_market) and total > total * .1:
    buy = buy_market()
    new_bought = {
      'minuteinfo': [],
      'trade_info': buy,
      'buy_price': 0, # to be define
      'sell_prices': {
        'up': last_price['Close'] * margin_up,
        'down': last_price['Close'] * margin_down 
      }
    }
    

  
def buy_market():
  pass

def sell_market(bought):
  pass


def should_buy(tail):
  '''return value 0 to 1'''
  return random.random()

def should_sell(bought, tail):
  '''return value 0 to 1'''
  return random.random()

def format_df(res):
  return {
      'Time': datetime.utcfromtimestamp(res['E']/1000).strftime('%Y-%m-%dT%H:%M:%SZ'),
      'Open': np.float64(res['k']['o']),
      'Close': np.float64(res['k']['c']),
      'High': np.float64(res['k']['h']),
      'Low': np.float64(res['k']['l']),
      'Trades': np.float64(res['k']['n']),
      'Open': np.float64(res['k']['v'])
    }

def append_data_from_stream(df, col):
    # TODO: solo tener una porcion del historial, y guardar los datos para uso posterior
    df = df.append(col, ignore_index=True)
    return df



async def main():
  client = await AsyncClient.create()
  await kline_listener(client)
  

if __name__ == "__main__":
  loop = asyncio.get_event_loop()
  loop.run_until_complete(main())