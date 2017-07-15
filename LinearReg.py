import pandas as pd
import quandl
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import style

style.use('ggplot')
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

df = quandl.get('WIKI/GOOGLE')
df.head()

df = df[['Adj. open', 'Adj. High', 'Adj. Low', 'Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'PCT_change', 'HL_PCT', 'Volume']]

forecast_col = 'Adj. Close'
df.fillna(-999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

print(df.head())

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_laterly = X[-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33)

classifier = LinearRegression()
# classifier = svm.SVR(kernel='poly')
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)

# print(accuracy)
forecast_set = classifier.predict(X_laterly)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1) + [i]]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Data')
plt.ylabel('Price')
plt.show()





















