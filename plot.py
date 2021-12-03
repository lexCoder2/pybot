import matplotlib.pyplot as plt
import pandas as pd


def graph(df, bar=True):
  if bar:
    plot_bar(df)
  else:
    plt.plot((df.close + df.open) / 2,linewidth=1, color='blue' )
  plt.show()

def plot_strategies(df):
  plt.figure(figsize=(16,8))
  plt.plot(df.max_high_wide, linewidth=1, color='orange')
  plt.plot(df.min_low_wide, linewidth=1, color='steelblue')
  # plt.plot(df.bollinger_u, linewidth=1, color='black')
  # plt.plot(df.bollinger_d, linewidth=1, color='black')
  # plt.plot(df.ema20, linewidth=1, color='tomato')
  plt.plot(df.sma200d, linewidth=1, color='green')
  plt.plot(df.supports, linewidth=1, color='black')

def plot_bar(prices: pd.DataFrame):
  width = .4
  width2 = .05

  up = prices[prices.close >= prices.open]
  down = prices[prices.close < prices.open]

  col1 = 'green'
  col2 = 'red'

  plt.bar(up.index, up.close - up.open, width, bottom=up.open, color=col1)
  plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color=col1)
  plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color=col1)

  plt.bar(down.index, down.close-down.open, width, bottom=down.open, color=col2)
  plt.bar(down.index, down.high-down.open, width2, bottom=down.open, color=col2)
  plt.bar(down.index, down.low-down.close, width2, bottom=down.close, color=col2)
  