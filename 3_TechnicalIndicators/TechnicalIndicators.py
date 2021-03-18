import numpy as np
import pandas as pd
import statsmodels.api as sm
from stocktrends import Renko

class TechnicalIndicators:
    
    @staticmethod
    def add_macd_signal(df: pd.DataFrame,
                        fast_ma_period: int = 12,
                        slow_ma_period: int = 26,
                        signal_period: int = 9) -> None:
        df['MA_Fast'] = df['Adj Close'].ewm(span=fast_ma_period, min_periods=fast_ma_period).mean()
        df['MA_Slow'] = df['Adj Close'].ewm(span=slow_ma_period, min_periods=slow_ma_period).mean()
        df['MACD'] = df['MA_Fast'] - df['MA_Slow']
        df['Signal'] = df['MACD'].ewm(span=signal_period, min_periods=signal_period).mean()
        df.drop(['MA_Fast', 'MA_Slow'], axis=1, inplace=True)
        df.dropna(inplace=True)

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 20, close_col_name: str = 'Adj Close') -> None:
        df['H-L'] = abs(df['High'] - df['Low'])
        df['H-PC'] = abs(df['High'] - df[close_col_name].shift(1))
        df['L-PC'] = abs(df['Low'] - df[close_col_name].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
        df['ATR'] = df['TR'].rolling(period).mean()
        df.drop(['H-L', 'H-PC', 'L-PC'], axis=1, inplace=True)
#         df.dropna(inplace=True)
        return df
        
    @staticmethod
    def add_bollinger(df: pd.DataFrame, period: int = 20) -> None:
        df['MA'] = df['Adj Close'].rolling(period).mean()
        df['TwoSD'] = 2 * df['Adj Close'].rolling(period).std(ddof=0)
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
        
    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> None:
        TechnicalIndicators.add_atr(df, period)
        df['DMPlus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                                df['High'] - df['High'].shift(1), 0)
        df['DMPlus'] = np.where(df['DMPlus'] < 0, 0, df['DMPlus'])
        df['DMMinus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                                 (df['Low'].shift(1) - df['Low']), 0)
        df['DMMinus'] = np.where(df['DMMinus'] < 0, 0, df['DMMinus'])
        TRn = []
        DMplusN = []
        DMminusN = []
        TR = df['TR'].values
        DMplus = df['DMPlus'].values
        DMminus = df['DMMinus'].values
        for i in range(len(df)):
            if i < period:
                TRn.append(np.NaN)
                DMplusN.append(np.NaN)
                DMminusN.append(np.NaN)
            elif i == period:
                TRn.append(df['TR'].rolling(period).sum().values[period])
                DMplusN.append(df['DMPlus'].rolling(period).sum().values[period])
                DMminusN.append(df['DMMinus'].rolling(period).sum().values[period])
            else:
                TRn.append(TRn[i-1] - (TRn[i-1]/period) + TR[i])
                DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/period) + DMplus[i])
                DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/period) + DMminus[i])
        df['TRn'] = np.array(TRn)
        df['DMplusN'] = np.array(DMplusN)
        df['DMminusN'] = np.array(DMminusN)
        df['DIplusN'] = 100*(df['DMplusN'] / df['TRn'])
        df['DIminusN'] = 100*(df['DMminusN'] / df['TRn'])
        df['DIdiff'] = abs(df['DIplusN'] - df['DIminusN'])
        df['DIsum'] = df['DIplusN'] + df['DIminusN']
        df['DX'] = 100*(df['DIdiff'] / df['DIsum'])
        ADX = []
        DX = df['DX'].values
        for i in range(len(df)):
            if i < 2 * period - 1:
                ADX.append(np.NaN)
            elif i == 2 * period - 1:
                ADX.append(df['DX'][i + 1 - period : i + 1].mean())
            else:
                ADX.append(((period-1)*ADX[i-1] + DX[i])/period)
        df['ADX'] = np.array(ADX)
        df.drop(['DMPlus', 'DMMinus', 'TRn', 'DMplusN', 'DMminusN', 'DIdiff', 'DIsum', 'DX', 'DIminusN', 'DIplusN'],
                axis=1, inplace=True)
        df.dropna(inplace=True)
    
    @staticmethod
    def add_obv(df: pd.DataFrame, close_col_name: str = 'Adj Close') -> None:
        df['daily_ret'] = df[close_col_name].pct_change()
        df['direction'] = np.where(df['daily_ret'] >= 0, 1, -1)
        df['direction'][0] = 0
        df['vol_adj'] = df['Volume'] * df['direction']
        df['obv'] = df['vol_adj'].cumsum()
        df.drop(['daily_ret', 'direction', 'vol_adj'], axis=1, inplace=True)
        df.dropna(inplace=True)
    
    @staticmethod
    def calc_slope(series: pd.Series, period: int = 5) -> np.ndarray: # 5 -> past 1 week
        series = (series - series.min()) / (series.max() - series.min())
        x = np.arange(0, len(series))
        x = (x - x.min()) / (x.max() - x.min())
        slopes = [0] * (period - 1)
        for i in range(period, len(series) + 1):
            y_scaled = series[i - period: i]
            x_scaled = x[:period]
            x_scaled = sm.add_constant(x_scaled)
            model = sm.OLS(y_scaled, x_scaled)
            results = model.fit()
            slopes.append(results.params[-1])
        slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
        return slope_angle
    
    @staticmethod
    def calc_slope_v2(series: pd.Series, period: int = 5) -> np.ndarray: # 5 -> past 1 week
        slopes = [0] * (period - 1)
        for i in range(period, len(series) + 1):
            y = series[i - period: i]
            x = np.arange(0, period)
            y_scaled = (y - y.min()) / (y.max() - y.min())
            x_scaled = (x - x.min()) / (x.max() - x.min())
            x_scaled = sm.add_constant(x_scaled)
            model = sm.OLS(y_scaled, x_scaled)
            results = model.fit()
            slopes.append(results.params[-1])
        slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
        return slope_angle
    
    @staticmethod
    def add_slope(df: pd.DataFrame, column_name: str, period: int = 5) -> None: # 5 -> past 1 week
        new_col_name = f'{column_name}_slope'
        df[new_col_name] = TechnicalIndicators.calc_slope(df[column_name], period)
    
    @staticmethod
    def slope_n_points(series: pd.Series, period: int = 5) -> np.ndarray:
        y_span = series.max() - series.min()
        x_span = 22
        slopes = np.zeros(period - 1)
        for i in range(period - 1, len(series)):
            y2 = series[i]
            y1 = series[i - n + 1]
            slope = ((y2 - y1) / y_span) / (period / x_span)
            slopes.append(slope)
        slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
        return slope_angle
    
    @classmethod
    def get_renko_df_yfinance(cls,
                              df_orig: pd.DataFrame,
                              max_brick_size: float = None,
                              extended: bool = False,
                              orig_close_col_name: str = 'Adj Close') -> pd.DataFrame:
        df = df_orig.copy()
        df.reset_index(inplace=True)
        df = df.iloc[:, [0, 1, 2, 3, 5, 6]]
        df.rename(columns = {"Date" : "date",
                             "High" : "high",
                             "Low" : "low",
                             "Open" : "open",
                             "Adj Close" : "close",
                             "Volume" : "volume"}, inplace = True)
        return cls._get_renko_df(df_orig, df, max_brick_size, extended, orig_close_col_name)
    
    @classmethod
    def get_renko_df_alpha_vantage(cls,
                                   df_orig: pd.DataFrame,
                                   max_brick_size: float = None,
                                   extended: bool = False,
                                   orig_close_col_name: str = 'Adj Close') -> pd.DataFrame:
        df = df_orig.copy()
        df.reset_index(inplace=True)
        df = df.iloc[:, [0, 1, 2, 3, 4, 5]]
        df.columns = ['date','open','high','low','close','volume']
        return cls._get_renko_df(df_orig, df, max_brick_size, extended, orig_close_col_name)
    
    @classmethod
    def _get_renko_df(cls,
                      df_orig: pd.DataFrame,
                      df_for_renko: pd.DataFrame,
                      max_brick_size: float,
                      extended: bool,
                      orig_close_col_name: str) -> pd.DataFrame:
        df2 = Renko(df_for_renko)
        cls.add_atr(df_orig, 120, orig_close_col_name)
        if max_brick_size is not None:
            df2.brick_size = max(max_brick_size, round(df_orig['ATR'][-1], 0))
        else:
            df2.brick_size = round(df_orig['ATR'][-1], 0)
        renko_df = df2.get_ohlc_data()
        if extended == False:
            return renko_df
        bar_num = np.where(renko_df['uptrend'] == True, 1, np.where(renko_df['uptrend'] == False, -1, 0))
        for i in range(1, len(bar_num)):
            if (bar_num[i] > 0 and bar_num[i-1] > 0) or (bar_num[i] < 0 and bar_num[i-1] < 0):
                bar_num[i] += bar_num[i-1]
        renko_df['bar_num'] = bar_num
        renko_df.drop_duplicates(subset='date', keep='last', inplace=True)
        return renko_df
