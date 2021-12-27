import pandas as pd
import numpy as np
import datetime as dt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from optimize_params import *

def cal_delta_bid_volume(row):
  if row["best_bid"] < row["best_bid"].shift(1):
      value = 0
  elif row["best_bid"] == row["best_bid"].shift(1):
      value = row["best_bid_volume"] - row["best_bid_volume"].shift(1)
  else:
      value = row["best_bid_volume"]
  return value
def parse_csv():
  print("---parse csv---")
  # dataframe = pd.read_csv("datasets/binance_book_snapshot_5_2019-11-01_BTCUSDT.csv", usecols=[2,4,5,6,7])
  usecols = ["ts", "bp1", "bs1", "ap1", "as1"]
  dataframe = pd.read_csv("datasets/BTCUSDT_S_DEPTH_20211028.csv", usecols=usecols)
  dataframe.rename(columns={"ts": "timestamp", "bp1": "best_bid", "bs1": "best_bid_volume", "ap1": "best_ask", "as1": "best_ask_volume"}, inplace=True)
  # dataframe.rename(columns={"bids[0].price": "best_bid", "bids[0].amount": "best_bid_volume", "asks[0].price": "best_ask", "asks[0].amount": "best_ask_volume"}, inplace=True)
  # print(dataframe.head(20))
  dataframe["date"] = pd.to_datetime(dataframe["timestamp"], unit="ms")
  # dataframe.to_csv("test.csv", encoding='utf-8', index=False)
  dataframe = dataframe.set_index(["date"])
  # dataframe["date"] = pd.to_datetime(dataframe["timestamp"], unit="ms")
  # To remove rows with duplicated indices
  dataframe = dataframe[~dataframe.index.duplicated(keep='first')]
  # print(dataframe.head())
  
  dataframe = dataframe.resample("5Min").ffill()
  print(dataframe.head(10))
  # drop NaNs for return
  dataframe.dropna(inplace=True)

  dataframe["bid_ask_spread"] = dataframe["best_ask"] - dataframe["best_bid"]
  dataframe["avg_price"] = (dataframe["best_bid"] + dataframe["best_ask"]) / 2
  dataframe["return"] = dataframe["avg_price"] - dataframe["avg_price"].shift(1)
  # drop NaNs for return
  dataframe.dropna(inplace=True)
  # delta bid volume calculation
  dataframe.loc[dataframe["best_bid"] < dataframe["best_bid"].shift(1), "delta_bid_volume"] = -dataframe["best_bid_volume"]
  dataframe.loc[dataframe["best_bid"] == dataframe["best_bid"].shift(1), "delta_bid_volume"] = dataframe["best_bid_volume"] - dataframe["best_bid_volume"].shift(1)
  dataframe.loc[dataframe["best_bid"] > dataframe["best_bid"].shift(1), "delta_bid_volume"] = dataframe["best_bid_volume"]
  # delta ask volume calculation
  dataframe.loc[dataframe["best_ask"] > dataframe["best_ask"].shift(1), "delta_ask_volume"] = -dataframe["best_ask_volume"]
  dataframe.loc[dataframe["best_ask"] == dataframe["best_ask"].shift(1), "delta_ask_volume"] = dataframe["best_ask_volume"] - dataframe["best_ask_volume"].shift(1)
  dataframe.loc[dataframe["best_ask"] < dataframe["best_ask"].shift(1), "delta_ask_volume"] = dataframe["best_ask_volume"]
  # volume order imbalance
  dataframe["voi"] = dataframe["delta_bid_volume"] - dataframe["delta_ask_volume"]
  # delta volume order imbalance ratio
  dataframe["delta_voi_r"] = (dataframe["delta_bid_volume"] - dataframe["delta_ask_volume"]) / (dataframe["delta_bid_volume"] + dataframe["delta_ask_volume"])
  # volume order imbalance ratio
  dataframe["voi_r"] = (dataframe["best_bid_volume"].shift(1) - dataframe["best_ask_volume"].shift(1)) / (dataframe["best_bid_volume"].shift(1) + dataframe["best_ask_volume"].shift(1))

  # drop NaNs
  dataframe.dropna(inplace=True)

  # # drop rows with voi is 0
  # dataframe = dataframe.drop(dataframe[(dataframe.voi == 0)].index)

  cols_to_select = ["voi", "delta_voi_r", "voi_r", "return"]
  dataframe = dataframe[cols_to_select].copy()

  # # Autocorelation plot
  # plt.acorr(dataframe["voi"], maxlags = 20)
  # plt.show()

  # # Return vs Spread
  # dataframe.plot(x="bid_ask_spread", y="return", style="o")
  # plt.title("Return vs Spread")
  # plt.show()

  # print(dataframe.head())

  X = dataframe.iloc[:, 0].values
  y = dataframe.iloc[:, -1].values

  # # Reciprocal
  # opt_param = optimize_reciprocal_params(X, y)
  # y_pred = reciprocal_func(X, *opt_param)

  ## Linear
  # a, b = optimize_linear_params(X, y)
  # y_pred = linear_func(X, a, b)
  # r = r2_score(y, y_pred)
  # print("==========r2_score===============")
  # print(r)

  # X = X.reshape(-1, 1)

  # poly_reg = PolynomialFeatures(degree = 4)
  # X_poly = poly_reg.fit_transform(X)
  # lin_reg = LinearRegression()
  # lin_reg.fit(X_poly, y)

  # y_pred = lin_reg.predict(X_poly)

  # # Comparing the Real Values with Predicted Values
  # df = pd.DataFrame({'Real Values':y, 'Predicted Values':y_pred})

  # # Visualising the Polynomial Regression results
  # X_grid = np.arange(min(X), max(X), 0.1)
  # X_grid = X_grid.reshape((len(X_grid), 1))
  # plt.scatter(X, y, color = 'red')
  # plt.scatter(X, y_pred, color = 'green')
  # plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color = 'black')
  # plt.title('Polynomial Regression')
  # plt.xlabel('VOI')
  # plt.ylabel('Return')
  # plt.show()

  # # Return vs Delta VOI_R
  # dataframe.plot(x="delta_voi_r", y="return", style="o")
  # plt.title("Return vs Delta VOI_R")
  # plt.axhline(0, color='black')
  # plt.axvline(0, color='black')
  # plt.show()

  # # Return vs VOI_R
  # dataframe.plot(x="voi_r", y="return", style="o")
  # plt.title("Return vs VOI_R")
  # plt.axhline(0, color='black')
  # plt.axvline(0, color='black')
  # plt.show()

  # plt.scatter(df["bid_ask_spread"], df["return"])
  # plt.show()

  # Return vs VOI
  dataframe.plot(x="voi", y="return", style="o")
  # plt.scatter(X, y_pred, color="red")
  plt.title("Return vs VOI")
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.show()

parse_csv()
