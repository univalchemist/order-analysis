import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# reciprocal function
def reciprocal_func(x, a, b, c):
    return b/(x + a) + c

def optimize_reciprocal_params(X, y):
  popt, pcov = curve_fit(reciprocal_func, X, y, bounds=([-100., 0.,  -100.], [100., 50., 100.]))
  print("***************optimzed params*****************")
  print(popt)
  return popt

# linear function
def linear_func(x, a, b):
  return a * x + b

def optimize_linear_params(X, y):
  slope, intercept, r, p, se = stats.linregress(X, y)
  print("***************optimzed params(a, b, r^2)*****************")
  print(slope, intercept, r**2)
  return slope, intercept

def compute_r2(y_true, y_predicted):
  sse = sum((y_true - y_predicted)**2)
  tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
  r2_score = 1 - (sse / tse)
  return r2_score, sse, tse