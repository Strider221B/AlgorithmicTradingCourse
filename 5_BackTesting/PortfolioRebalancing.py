import numpy as np
import pandas as pd
from typing import Dict, List

class PortfolioRebalancing:
    
    @classmethod
    def get_monthly_return(cls, ohlc: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return_df = pd.DataFrame()
        for ticker in ohlc:
            monthly_return = ohlc[ticker]['Adj Close'].pct_change().values
            return_df[ticker] = monthly_return
        return_df.index = ohlc[return_df.columns[0]].index
        return return_df
    
    @classmethod
    def cumulative_portfolio_returns(cls,
                                     df: pd.DataFrame,
                                     no_of_stocks_in_portfolio: int,
                                     no_of_stocks_to_swap: int) -> (np.ndarray,
                                                                    List[List[str]]):
        portfolio = []
        monthly_returns = [0]
        portfolio_history = []
        for i in range(len(df)):
            if len(portfolio) > 0:
                monthly_returns.append(df[portfolio].iloc[i, :].mean())
                bad_stocks = df[portfolio].iloc[i, :].sort_values(ascending=True)[:no_of_stocks_to_swap].index.values
                portfolio = [t for t in portfolio if t not in bad_stocks]
            fill = no_of_stocks_in_portfolio - len(portfolio)
            new_picks = df.iloc[i, :].sort_values(ascending=False)[:fill].index.values
            portfolio.extend(new_picks)
            portfolio_history.append(portfolio)
        monthly_returns = np.array(monthly_returns)
        return monthly_returns, portfolio_history
