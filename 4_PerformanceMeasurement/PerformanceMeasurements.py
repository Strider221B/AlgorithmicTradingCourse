import numpy as np
import pandas as pd

class PerformanceMeasurements:
    
    @staticmethod
    def get_cagr_for(df: pd.DataFrame) -> float:
        daily_ret = df['Adj Close'].pct_change().values
        daily_ret[0] = 0 # pct_change makes the first value nan which messes up numpy calc, but not pandas calc
        cum_return = (1 + daily_ret).cumprod()
        no_of_years = len(df) / 252
        cagr = ((cum_return[-1])**(1/no_of_years)) - 1
        return cagr
    
    @staticmethod
    def get_annualized_volatility(df: pd.DataFrame) -> float:
        daily_ret = df['Adj Close'].pct_change().values
        daily_ret = daily_ret[1:]
        volaility  = daily_ret.std() * np.sqrt(252)
        return volaility
