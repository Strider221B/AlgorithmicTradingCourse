import numpy as np
import pandas as pd

class TechnicalIndicators:
    
    @staticmethod
    def add_macd_signal(df: pd.DataFrame, fast_ma_period: int = 12, slow_ma_period: int = 26, signal_period: int = 9) -> None:
        df['MA_Fast'] = df['Adj Close'].ewm(span=fast_ma_period, min_periods=fast_ma_period).mean()
        df['MA_Slow'] = df['Adj Close'].ewm(span=slow_ma_period, min_periods=slow_ma_period).mean()
        df['MACD'] = df['MA_Fast'] - df['MA_Slow']
        df['Signal'] = df['MACD'].ewm(span=signal_period, min_periods=signal_period).mean()
        df.drop(['MA_Fast', 'MA_Slow'], axis=1, inplace=True)
        df.dropna(inplace=True)

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 20) -> None:
        df['H-L'] = abs(df['High'] - df['Low'])
        df['H-PC'] = abs(df['High'] - df['Adj Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Adj Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
        df['ATR'] = df['TR'].rolling(period).mean()
        df.drop(['H-L', 'H-PC', 'L-PC'], axis=1, inplace=True)
        df.dropna(inplace=True)
        
    @staticmethod
    def add_bollinger(df: pd.DataFrame, period: int = 20) -> None:
        df['MA'] = df['Adj Close'].rolling(period).mean()
        df['TwoSD'] = 2 * df['MA'].rolling(period).std()
        df['BB_up'] = df['MA'] + df['TwoSD']
        df['BB_dn'] = df['MA'] - df['TwoSD']
        df['BB_Width'] = df['BB_up'] - df['BB_dn']
        df.drop(['TwoSD'], axis=1, inplace=True)
        df.dropna(inplace=True)
       
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> None:
        df['Delta'] = df['Adj Close'] - df['Adj Close'].shift(1)
        df['Gain'] = np.where(df['Delta'] >=0, df['Delta'], 0)
        df['Loss'] = np.where(df['Delta'] <0, abs(df['Delta']), 0)
        avg_gain = []
        avg_loss = []
        gain = df['Gain'].values
        loss = df['Loss'].values
        for i in range(len(df)):
            if i < period:
                avg_gain.append(np.NaN)
                avg_loss.append(np.NaN)
            elif i == period:
                avg_gain.append(df['Gain'].rolling(period).mean().tolist()[period])
                avg_loss.append(df['Loss'].rolling(period).mean().tolist()[period])
            else:
                avg_gain.append(((period - 1) * avg_gain[i - 1] + gain[i])/period)
                avg_loss.append(((period - 1) * avg_loss[i - 1] + loss[i])/period)
        avg_gain = np.array(avg_gain)
        avg_loss = np.array(avg_loss)
        df['RS'] = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + df['RS']))
        df.drop(['Delta', 'Gain', 'Loss'], axis=1, inplace=True)
        df.dropna(inplace=True)