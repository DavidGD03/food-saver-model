import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM
import argparse
import os

parser = argparse.ArgumentParser(description='train food saver')
parser.add_argument('--data_url', type=str, default="./data/train.csv", help='path where the dataset is saved')
parser.add_argument('--train_url', type=str, default="./export", help='path where the model is saved')
args = parser.parse_args()


#Function to calculate date based on the given number
def calculate_date(row):
    base_date = datetime(2023, 3, 1)  # March 1st, 2023
    delta = timedelta(days=row['week'] - 1)
    return base_date + delta

df_multivariable = pd.read_csv(args.data_url)
meals_list = [1878, 1754, 1558, 2581, 1962, 2104, 1571, 2444]
df_multivariable = df_multivariable[df_multivariable['meal_id'].isin(meals_list)]
# Apply the function to create the 'Date' column
df_multivariable['date'] = df_multivariable.apply(calculate_date, axis=1)
df_multivariable.set_index('date',inplace=True)
df_real_data = df_multivariable.copy(deep=True)

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data=X_scaler.fit_transform(df_multivariable.drop(columns=['num_orders']))
Y_data = Y_scaler.fit_transform(df_multivariable[['num_orders']])

df_multivariable[['id', 'week',	'center_id', 'meal_id', 'checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured']] = X_data
df_multivariable[['num_orders']] = Y_data

train_MRNN_sc=df_multivariable['2023-03-01':'2023-06-30']
test_MRNN_sc=df_multivariable['2023-07-01':]

train_MRNN_scN=train_MRNN_sc.to_numpy()
test_MRNN_scN=test_MRNN_sc.to_numpy()
n_input = int(7)
n_features = 9

generator = TimeseriesGenerator(train_MRNN_scN, train_MRNN_scN, length=n_input, batch_size=1)   

# define model
model = Sequential()
model.add(LSTM(64, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
model.add(LSTM(128, return_sequences=True,activation='relu'))
model.add(LSTM(256, return_sequences=True,activation='relu'))
model.add(LSTM(128, return_sequences=True,activation='relu'))
model.add(LSTM(64, return_sequences=True,activation='relu'))
model.add(LSTM(n_features)) 

model.compile(optimizer='adam', loss='mse')
history=model.fit(generator,epochs=100)

model.save(os.path.join(args.train_url, 'model'))