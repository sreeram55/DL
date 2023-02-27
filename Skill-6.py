import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.layers import Normalization, Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

df = pd.read_csv('Data/car_sales_linear.csv')
print(df.head())

print(df.isnull().sum())

print(df.describe())


print('Shape of dataset:', df.shape)

corr = df.corr()
print(corr)
sns.heatmap(corr, annot=True)

sns.pairplot(df)

sns.displot(df)

df = df.ffill()

print(df.isnull().sum())

X = df.Engine_size
y = df.sales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

normalizer = Normalization(input_shape=[1, ], axis=None)
normalizer.adapt(X)

model = Sequential([
    normalizer,
    Dense(1)
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_absolute_error')

callback = EarlyStopping(monitor='loss', patience=3)

model.fit(X_train, y_train, epochs=100, verbose=0, callbacks=[callback], validation_data=(X_test, y_test))

y_predicted = model.predict(X_test)
print(y_predicted)
