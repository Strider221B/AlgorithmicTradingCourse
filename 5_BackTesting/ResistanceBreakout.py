import numpy as np
import pandas as pd
from typing import Dict, List

class ResistanceBreakout:
    
    @classmethod
    def calculate_returns(cls,
                          ohlc_dict: Dict[str, pd.DataFrame]):
        tickers_signal: Dict[str, str] = {}
        tickers_return: Dict[str, List[float]] = {}
        for ticker in ohlc_dict:
            tickers_signal[ticker] = ''
            tickers_return[ticker] = [0]
        for ticker in ohlc_dict:
            ohlc = ohlc_dict[ticker]
            for i in range(1, len(ohlc)):
                vol_greater_than_threshold = ohlc['Volume'][i] > 1.5 * ohlc['roll_max_vol'][i-1]
                low_less_than_roll_min_cp = ohlc['Low'][i] <= ohlc['roll_min_cp'][i]
                high_greater_than_roll_max_cp = ohlc['High'][i] >= ohlc['roll_max_cp'][i]
                if tickers_signal[ticker] == '':
                    tickers_return[ticker].append(0)
                    if vol_greater_than_threshold:
                        if high_greater_than_roll_max_cp:
                            tickers_signal[ticker] = 'buy'
                        elif low_less_than_roll_min_cp:
                            tickers_signal[ticker] = 'sell'
                elif tickers_signal[ticker] == 'buy':
                    if ohlc['Low'][i] < ohlc['Close'][i-1] - ohlc['ATR'][i-1]:
                        tickers_signal[ticker] = ''
                        tickers_return[ticker].append(((ohlc['Close'][i-1] - ohlc['ATR']) / ohlc_dict[ticker]['Close'][i-1])-1)
                    else:
                        tickers_return[ticker].append((ohlc['Close'][i]/ohlc['Close'][i-1])-1)
                        if low_less_than_roll_min_cp and vol_greater_than_threshold:
                            tickers_signal[ticker] = 'sell'
                else:
                    if ohlc['High'][i] < ohlc['Close'][i-1] + ohlc['ATR'][i-1]:
                        tickers_signal[ticker] = ''
                        tickers_return[ticker].append((ohlc['Close'][i-1]/(ohlc['Close'][i-1] + ohlc['ATR'][i-1])) - 1)
                    else:
                        tickers_return[ticker].append((ohlc['Close'][i-1] / ohlc['Close'][i]) - 1)
                        if high_greater_than_roll_max_cp and vol_greater_than_threshold:
                            tickers_signal[ticker] = 'buy'
            ohlc['ret'] = np.array(tickers_return[ticker])
