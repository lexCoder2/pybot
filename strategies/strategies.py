import math
import pandas as pd

class Strategies:

  def __init__(self, df) -> None:
    self.df: pd.DataFrame = df    
    pass

  def _period(self, period):
    if period == 'days':
      return 1
    elif period == '4hour':
      return 240
    elif period == 'day':
      return 1440
    elif period == '1hour':
      return 60

  def calc_all(self, period='minute'):
    s = self._period(period)
    bollinger =  self.calc_bollinger(s * 20)
    osc_stocast = self.calc_osc_esotc(s * 14)
    return ({
      'bollinger_u': bollinger['BOLU'],
      'bollinger_d': bollinger['BOLD'],
      'ema': self.calc_ema(s),
      'ema20': self.calc_ema(s * 20),
      'ema55': self.calc_ema(s * 55),
      'osc_estoc_k': osc_stocast['K'],
      'osc_estoc_D': osc_stocast['D'],
      'rsi': self.calc_rsi(),
      'sma': self.calc_sma(),
      'sma200d': self.calc_sma(self._period('day') * 200),
      'supports': self.supports(s * 20)
    })

  def calc_rsi(self):
    delta = self.df['close'].diff()
    up = delta.clip(lower = 0)
    down = -1 * delta.clip(upper = 0)
    ema_up = up.ewm(com = 13, adjust = False).mean()
    ema_down = down.ewm(com = 13, adjust = False).mean()
    rs = ema_up / ema_down
    self.rsi = 100 - (100/(1 + rs))
    return self.rsi


  def calc_sma(self, window = 20):
    self.sma = self.df['close'].rolling(window=window, min_periods=1).mean()
    return self.sma


  def supports(self, window = 20):
    self.supports = (self.df['close'].rolling(window=window, min_periods=1).mean() / (self.df.high.mean() /20)).round() * (self.df.high.mean()/20)
    return self.supports
  
  def calc_ema(self, window = 20):

    sma = self.df['close'].rolling(window=window, min_periods=window).mean()[:window]
    rest = self.df['close'][window:]
    self.ema = pd.concat([sma, rest]).ewm(span=window, adjust=False).mean()
    return self.ema

  def calc_osc_esotc(self, window = 14):
    hig14 = self.df['high'].rolling(window).max()
    low14 = self.df['low'].rolling(window).min()
    k = (self.df['close'] - low14) * 100 / (hig14 - low14)
    self.estoc = pd.DataFrame({
      'K': k,
      'D': k.rolling(3).mean()
    })
    return self.estoc

  def calc_bollinger(self, window = 20):
    tp = (self.df['close'] + self.df['low'] + self.df['high']) / 3
    std = tp.rolling(window).std(ddof=0)
    self.mean20 = tp.rolling(window).mean()
    self.bollinger = pd.DataFrame({
        'BOLU': self.mean20 + 2 * std,
        'BOLD': self.mean20 - 2 * std
    })
  
    return self.bollinger

  def calc_adx(self):

    def getCDM(df):
        dmpos = df["high"][-1] - df["high"][-2]
        dmneg = df["low"][-2] - df["low"][-1]
        if dmpos > dmneg:
            return dmpos
        else:
            return dmneg 

    def getDMnTR(df):
        DMpos = []
        DMneg = []
        TRarr = []
        n = round(len(df)/14)
        idx = n
        while n <= (len(df)):
            dmpos = df["high"][n-1] - df["high"][n-2]
            dmneg = df["low"][n-2] - df["low"][n-1]
                
            DMpos.append(dmpos)
            DMneg.append(dmneg)
        
            a1 = df["high"][n-1] - df["high"][n-2]
            a2 = df["high"][n-1] - df["close"][n-2]
            a3 = df["low"][n-1] - df["close"][n-2]
            TRarr.append(max(a1,a2,a3))

            n = idx + n
    
        return DMpos, DMneg, TRarr

    def getDI(df):
        DMpos, DMneg, TR = getDMnTR(df)
        CDM = getCDM(df)
        POSsmooth = (sum(DMpos) - sum(DMpos)/len(DMpos) + CDM)
        NEGsmooth = (sum(DMneg) - sum(DMneg)/len(DMneg) + CDM)
        
        DIpos = (POSsmooth / (sum(TR)/len(TR))) *100
        DIneg = (NEGsmooth / (sum(TR)/len(TR))) *100

        return DIpos, DIneg

    def getADX(df):
        DIpos, DIneg = getDI(df)
        dx = (abs(DIpos- DIneg) / abs(DIpos + DIneg)) * 100    
        ADX = dx/14
        return ADX

    self.adx = getADX(self.df)
    return(self.adx)



