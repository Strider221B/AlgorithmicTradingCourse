import numpy as np
import pandas as pd

class PerformanceMeasurements:
    
    @classmethod
    def get_cagr_for(cls, df: pd.DataFrame) -> float:
        daily_ret = df['Adj Close'].pct_change().values
        daily_ret[0] = 0 # pct_change makes the first value nan which messes up numpy calc, but not pandas calc
        return cls.get_cagr_for_returns(daily_ret, 252)
    
    @classmethod
    def get_cagr_for_returns(cls,
                             returns_data: np.ndarray,
                             no_of_datapoints_in_year: int) -> float:
        cum_return = (1 + returns_data).cumprod()
        no_of_years = len(returns_data) / no_of_datapoints_in_year
        cagr = ((cum_return[-1])**(1/no_of_years)) - 1
        return cagr
    
    @classmethod
    def get_annualized_volatility(cls, df: pd.DataFrame) -> float:
        daily_ret = df['Adj Close'].pct_change().values
        daily_ret = daily_ret[1:]
        return cls.get_annualized_volatility_for_returns(daily_ret, 252)
    
    @classmethod
    def get_annualized_volatility_for_returns(cls,
                                              returns_data: np.ndarray,
                                              no_of_datapoints_in_year: int) -> float:
        volaility  = returns_data.std() * np.sqrt(no_of_datapoints_in_year)
        return volaility
    
    @classmethod
    def get_sharpe_ratio(cls, df: pd.DataFrame, risk_free_rate: float) -> float:
        '''
        risk_free_rate: For India it could be the FD rate, for USA it could be the govt bond rate.
        '''
        sr = (cls.get_cagr_for(df) - risk_free_rate) / cls.get_annualized_volatility(df)
        return sr
    
    @classmethod
    def get_sharpe_ratio_for_returns(cls,
                                     returns_data: np.ndarray,
                                     no_of_datapoints_in_year: int,
                                     risk_free_rate: float) -> float:
        '''
        risk_free_rate: For India it could be the FD rate, for USA it could be the govt bond rate.
        '''
        sr = (cls.get_cagr_for_returns(returns_data,
                                       no_of_datapoints_in_year) - risk_free_rate) / cls.get_annualized_volatility_for_returns(returns_data, no_of_datapoints_in_year)
        return sr
    
    @classmethod
    def get_sortino_ratio(cls, df: pd.DataFrame, risk_free_rate: float) -> float:
        '''
        risk_free_rate: For India it could be the FD rate, for USA it could be the govt bond rate.
        '''
        daily_ret = df['Adj Close'].pct_change().values
        neg_returns = np.where(daily_ret < 0, daily_ret, 0)
        neg_volatility = neg_returns.std() * 252**0.5
        sr = (cls.get_cagr_for(df) - risk_free_rate) / neg_volatility
        return sr
    
    @classmethod
    def get_max_drawdown(cls, df: pd.DataFrame) -> float:
        daily_ret = df['Adj Close'].pct_change().values
        daily_ret[0] = 0
        return cls.get_max_drawdown_for_returns(daily_ret)
    
    @classmethod
    def get_max_drawdown_for_returns(cls,
                                     returns_data: np.ndarray) -> float:
        cum_return = (1 + returns_data).cumprod()
        cum_roll_max = np.maximum.accumulate(cum_return)
        draw_down = cum_roll_max - cum_return
        draw_down_pct = draw_down / cum_roll_max
        max_dd = draw_down_pct.max()
        return max_dd
    
    @classmethod
    def get_calmar(cls, df: pd.DataFrame) -> float:
        calmar = cls.get_cagr_for(df) / cls.get_max_drawdown(df)
        return calmar
