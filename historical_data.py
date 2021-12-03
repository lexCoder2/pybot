import pandas as pd
from pandas.api.types import is_numeric_dtype

class HistoricalData:

  def __init__(self, csv, limit_rows=0, coin_name='BTC') -> None:
    self.columns = [ 'open', 'close', 'high', 'low', 'volume_coin']
    self.optional_columns = [ 'volume_currency']
    self.csv_file = csv
    self.coin_name = coin_name
    self.get_data(limit_rows)
    self.estandarize()
    self.clean_data()
    self.data_day = None

  def get_data(self, limit_rows):
    
    csv_read = pd.read_csv(self.csv_file)
    if(limit_rows < csv_read.shape[0] and limit_rows > 0):
      self.df = csv_read.tail(limit_rows)
    else:
      self.df = csv_read

  def estandarize(self):

    def change_names(name:str):
      name = name.lower()
      if name in ['timestamp', 't']:
        return 'time'
      elif name in ['volume', 'volume_(' + self.coin_name.lower() + ')']:
        return 'volume_coin'
      elif name in ['volume_(usdt)', 'volume_(usd)', 'volume_(currency)']:
        return 'volume_currency'
      else:
        return name

    self.df = self.df.rename(columns = change_names)
    for col in self.df.columns:
      if (col not in self.columns + self.optional_columns + ['time']):
        self.df.drop(col, axis=1, inplace=True)

  def data_minutes(self):
    return self.df

  def _calc_date(self):
    df = pd.DataFrame(columns=self.columns)
    df.open = self.df.open.resample('1D').first()
    df.close = self.df.close.resample('1D').last()
    df.high = self.df.high.resample('1D').max()
    df.low = self.df.low.resample('1D').min()
    df.volume_coin = self.df.volume_coin.resample('1D').sum()
    if 'volume_currency'in self.df:
      df['volume_currency'] = self.df.volume_currency.resample('1D').sum()
    self.data_day = df
    return self.data_day
    
  def _calc_windows(self, minutes_per_windows):
      data = pd.DataFrame(columns=self.columns)
      for i in range(minutes_per_windows, self.df.shape[0] - minutes_per_windows, minutes_per_windows):
        a = [ 
              self.df.iloc[i - minutes_per_windows]['open'],
              self.df.iloc[i - minutes_per_windows : i]['high'].max(),
              self.df.iloc[i - minutes_per_windows : i]['low'].min(),
              self.df.iloc[i]['close'],
              self.df.iloc[minutes_per_windows : i]['volume_coin'].sum(),
            ]
        if 'volume_currency' in self.df:
          a.append(self.df.iloc[minutes_per_windows : i]['volume_currency'].sum())
        data = data.append(pd.Series(a,
            name = self.df.iloc[i].name,
            index = self.df.columns
            ))
      return data


  def get_data_day(self):
    if not self.data_day: 
      self.data_day = self._calc_date()
    return self.data_day

  def get_data_hour(self):
    minutes_hour = 60
    if not self.data_hour: 
      self.data_hour = self._calc_windows(minutes_hour)
    return self.data_hour

  def clean_data(self):
    if ('weighted_price' in self.df.columns):
      self.df = self.df.interpolate().drop('weighted_price', axis=1)
    self.df.dropna(how="all", subset=['open' ,'high','low','close','volume_coin'], inplace=True)
    if is_numeric_dtype(self.df['time']):
      unit = 'ms' if len(str(self.df.time.iloc[0])) == 13 else 's'
        
    self.df['time'] = pd.to_datetime(self.df['time'], unit=unit)
    self.df.set_index('time', inplace=True)
    
