# module5_factor_regression.py
import numpy as np
import pandas as pd

class CrossSectionRegression:
    def __init__(self, factor_cols):
        self.factor_cols = factor_cols

    def fit_one_day(self, df: pd.DataFrame):
        """
        df: columns = [y] + factor_cols
        """
        df = df.dropna()
        if len(df) < len(self.factor_cols) + 5:
            return None, None

        y = df["y"].values
        X = df[self.factor_cols].values
        X = np.c_[np.ones(len(X)), X]  # intercept

        try:
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None, None

        alpha = coef[0]
        beta = pd.Series(coef[1:], index=self.factor_cols)

        return alpha, beta
