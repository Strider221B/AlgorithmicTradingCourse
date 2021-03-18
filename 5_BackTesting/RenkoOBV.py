import numpy as np
import pandas as pd
from typing import Dict, List

class RenkoOBV:
    
    @classmethod
    def calculate_returns(cls,
                          ohlc_dict: Dict[str, pd.DataFrame],
                          close_col_name: str = 'Adj Close'):
        tickers_signal: Dict[str, str] = {}
        tickers_return: Dict[str, List[float]] = {}
        for ticker in ohlc_dict:
            tickers_signal[ticker] = ''
            tickers_return[ticker] = [0]
        for ticker in ohlc_dict:
            ohlc = ohlc_dict[ticker]
            for i in range(1, len(ohlc)):
                renko_gte_two = ohlc['bar_num'].iloc[i] >= 2 # gte -> greater than equal two
                obv_slope_greater_than_thirty = ohlc['obv_slope'].iloc[i] > 30
                renko_lte_minus_two = ohlc['bar_num'].iloc[i] <= -2 
                obv_slope_less_than_minus_thirty = ohlc['obv_slope'].iloc[i] < -30
                if tickers_signal[ticker] == '':
                    tickers_return[ticker].append(0)
                    if renko_gte_two and obv_slope_greater_than_thirty:
                        tickers_signal[ticker] = 'buy'
                    elif renko_lte_minus_two and obv_slope_less_than_minus_thirty:
                        tickers_signal[ticker] = 'sell'
                elif tickers_signal[ticker] == 'buy':
                    tickers_return[ticker].append((ohlc[close_col_name].iloc[i] / ohlc[close_col_name].iloc[i-1]) - 1)
                    if renko_lte_minus_two and obv_slope_less_than_minus_thirty:
                        tickers_signal[ticker] = 'sell'
                    elif ohlc['bar_num'].iloc[i] < 2:
                        tickers_signal[ticker] = ''
                else:
                    tickers_return[ticker].append((ohlc[close_col_name].iloc[i-1] / ohlc[close_col_name].iloc[i]) - 1)
                    if renko_gte_two and obv_slope_greater_than_thirty:
                        tickers_signal[ticker] = 'buy'
                    elif ohlc['bar_num'].iloc[i] > -2:
                        tickers_signal[ticker] = ''
            ohlc['ret'] = np.array(tickers_return[ticker])
