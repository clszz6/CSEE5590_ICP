import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

"""
Question 1 Start
"""
train = pd.read_csv('train.csv')

plt.scatter(train.SalePrice, train.GarageArea, alpha=.75, color='b')
plt.show()

data = train.select_dtypes(include=[np.number]).interpolate().dropna()

data = data[np.abs(data.SalePrice-data.SalePrice.mean()) <= (2 * data.SalePrice.std())]
data = data[np.abs(data.GarageArea-data.GarageArea.mean()) <= (2 * data.GarageArea.std())]

plt.scatter(data.SalePrice, data.GarageArea, alpha=.75, color='b')
plt.show()

"""
Question 1 end
"""



"""
Question 2 Start
"""

weather = pd.read_csv('weatherHistory.csv')

#Next, we'll check for skewness
print ("Skew is:", weather.Temperature.skew())
plt.hist(weather.Temperature, color='blue')
plt.show()

# target = np.log(weather.Temperature)
# print ("Skew is:", target.skew())
# plt.hist(target, color='blue')
# plt.show()

#Working with Numeric Features
numeric_features = weather.select_dtypes(include=[np.number])

corr = numeric_features.corr()

##handling missing value
data = weather.select_dtypes(include=[np.number]).interpolate().dropna()
# print(sum(data.isnull().sum() != 0))

##Build a linear model
y = weather.Temperature
X = weather[["ApparentTemperature", "Humidity"]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

##visualize
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Linear Regression Model')
plt.show()


"""
Question 2 end
"""
