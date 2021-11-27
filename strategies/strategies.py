import pandas as pd

class Strategies:

  def __init__(self, df) -> None:
    self.df = df    
    pass

  def calc_all(self):
    return ([
      # self.calc_adx(),
      self.calc_bollinger(),
      self.calc_ema(),
      self.calc_osc_esotc(),
      self.calc_rsi(),
      self.calc_sma()
    ])

  def calc_rsi(self):
    delta = self.df['Close'].diff()
    up = delta.clip(lower = 0)
    down = -1 * delta.clip(upper = 0)
    ema_up = up.ewm(com = 13, adjust = False).mean()
    ema_down = down.ewm(com = 13, adjust = False).mean()
    rs = ema_up / ema_down
    self.rsi = 100 - (100/(1 + rs))
    return self.rsi


  def calc_sma(self, window = 20):
    self.sma = self.df['Close'].rolling(window=window, min_periods=1).mean()
    return self.sma

  def calc_ema(self, window = 20):
    sma = self.df['Close'].rolling(window=window, min_periods=window).mean()[:window]
    rest = self.df['Close'][window:]
    self.ema = pd.concat([sma, rest]).ewm(span=window, adjust=False).mean()
    return self.ema

  def calc_osc_esotc(self):
    hig14 = self.df['High'].rolling(14).max()
    low14 = self.df['Low'].rolling(14).min()
    k = (self.df['Close'] - low14) * 100 / (hig14 - low14)
    self.estoc = pd.DataFrame({
      'K': k,
      'D': k.rolling(3).mean()
    })
    return self.estoc

  def calc_bollinger(self):
    tp = (self.df['Close'] + self.df['Low'] + self.df['High']) / 3
    std = tp.rolling(20).std(ddof=0)
    self.mean20 = tp.rolling(20).mean()
    self.bollinger = pd.DataFrame({
        'BOLU': self.mean20 + 2 * std,
        'BOLD': self.mean20 - 2 * std
    })
  
    return self.bollinger

  def calc_adx(self):

    def getCDM(df):
        dmpos = df["High"][-1] - df["High"][-2]
        dmneg = df["Low"][-2] - df["Low"][-1]
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
            dmpos = df["High"][n-1] - df["High"][n-2]
            dmneg = df["Low"][n-2] - df["Low"][n-1]
                
            DMpos.append(dmpos)
            DMneg.append(dmneg)
        
            a1 = df["High"][n-1] - df["High"][n-2]
            a2 = df["High"][n-1] - df["Close"][n-2]
            a3 = df["Low"][n-1] - df["Close"][n-2]
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



