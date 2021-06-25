import pandas as pd
import numpy as np
import yfinance as yf
import config


def rsi(data, period):
    """Momentum oscillator that measures the speed
        and change of price movements, from [0, 100]
    Args:
        data [float]: pandas.DataFrame of SPY
        period [int]: number of days
    Returns:
        [float]: RSI oscillates between zero and 100 (overbought, oversold)
    """
    diff = data.diff(1).dropna()
    up = 0 * diff
    down = 0 * diff
    # up change is equal to the pos/neg difference, otherwise equal to zero
    up[diff > 0] = diff[diff > 0]
    down[diff < 0] = diff[diff < 0]

    # Use exponential moving average
    ema_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    ema_down = down.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rs = abs(ema_up / ema_down)
    return 100 - 100 / (1 + rs)


def moving_average50(data, period):
    """
    Args:
        data [float]: pandas.DataFrame of SPY
        period [int]: number of period days (50)
    Returns:
        [float]: set average price of a security over a set 50 period
    """
    ma = data.rolling(window=period).mean()
    return ma


def moving_average200(data, period):
    """
    Args:
        data [float]: pandas.DataFrame of SPY
        period [float]: number of period days (200)

    Returns:
        [float]: set average price of a security over a set 200 period
    """
    ma = data.rolling(window=period).mean()
    return ma


def stochastic_oscillator(data, period):
    """A momentum indicator comparing a particular closing
        price of a security to a range of its prices over
        a certain period of time
    Args:
        data [float]: pandas.DataFrame of SPY
        period [int]: The lowest/highest price of 14 trading days
    Returns:
        [float]: percentage based [0,100]
    """
    data["14-high"] = data["High"].rolling(period).max()
    data["14-low"] = data["Low"].rolling(period).min()
    data["%K"] = (
        (data["Adj Close"] - data["14-low"]) * 100 / (data["14-high"] - data["14-low"])
    )
    return data["%K"].rolling(3).mean()


def macd(data, short_period, long_period):
    """Trend-following momentum indicator that
        shows the relationship between two moving
        averages of a security’s price.
    Args:
        data [float]: pandas.DataFrame of SPY
        short_period [int]: 12-period EMA
        long_period [int]: 26-period EMA

    Returns:
        [float]: Subtracting the 26-period (EMA) from the 12-period EMA
    """
    exp1 = data.ewm(span=short_period, adjust=False).mean()
    exp2 = data.ewm(span=long_period, adjust=False).mean()
    return exp1 - exp2


def signal_macd(data, period):
    """
    Args:
        data [float]: pandas.DataFrame of SPY
        period [int]: 9-day EMA

    Returns:
        [float]: 9-day EMA of the MACD Line.
    """
    return macd(data, 12, 26).ewm(span=period, adjust=False).mean()


def log_returns(data):
    """
    Args:
        data [float]: pandas.DataFrame of SPY
    Returns:
        [float]: Adj-Close Log-Returns, normalized values
    """
    return np.log(data["Adj Close"]) - np.log(data["Adj Close"].shift(1))


def predict(data):
    """Target Value as binary (1=up_day, 0=down_day)
    Args:
        data [float]: pandas.DataFrame of SPY
    Returns:
        [int]: as bianry value [0, 1]
    """
    return data.transform(lambda x: x.shift(1) < x) * 1


if __name__ == "__main__":
    df = yf.download("SPY", start="2009-01-01", end="2021-06-22")
    df["RSI"] = rsi(df["Adj Close"], 14)
    df["50MA"] = moving_average50(df["Adj Close"], 50)
    df["200MA"] = moving_average200(df["Adj Close"], 200)
    df["SC"] = stochastic_oscillator(df, 14)
    df["MACD"] = macd(df["Adj Close"], 12, 16)
    df["Log_Returns"] = log_returns(df)
    df["Signal_MACD"] = signal_macd(df["Adj Close"], 9)
    df["Target"] = predict(df["Adj Close"])
    df = df.drop(["Open", "High", "Low", "Close", "Volume", "Adj Close"], axis=1)
    df = df[200:]
    if df.isnull().sum().any() == False:
        print("Data is Clean")
        df.to_csv(config.TRAINING_FILE, index="Date")
    else:
        print("Data is not Clean")
