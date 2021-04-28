import sys
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)

@app.route('/predict', methods=['POST','GET'])
def train():
    # try:
        jsonfile = request.get_json()
        df = pd.read_json(jsonfile,orient='records')
        df['time_period_start'] = pd.to_datetime(df['time_period_start'])
        df.set_axis(df['time_period_start'], inplace=True)
        df.dropna(inplace=True)

        close_data = df['price_close'].values
        open_data = df['price_open'].values
        open_data = open_data.reshape((-1,1))

        split_percent = 0.80
        split = int(split_percent*len(open_data))

        open_train = open_data[:split]
        open_test = open_data[split:]

        date_train = df['time_period_start'][:split]
        date_test = df['time_period_start'][split:]

        look_back = 30
        train_generator_open = TimeseriesGenerator(open_train, open_train, length=look_back, batch_size=20)     
        test_generator_open = TimeseriesGenerator(open_test, open_test, length=look_back, batch_size=1)

        model = Sequential()
        model.add(
            LSTM(10,
                activation='relu',
                input_shape=(look_back,1))
        )
        model.add(Dropout(0.3))
        model.add(Dense(1))

        opt = keras.optimizers.RMSprop(learning_rate=0.01)
        model.compile(
        optimizer = opt,
        loss = 'msle'
        )

    #     checkpointer = ModelCheckpoint('USD_JPY_open.h5', verbose=1, save_best_only=True)
        callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=3, monitor='loss'),
                tf.keras.callbacks.TensorBoard(log_dir='logs')]

        model.fit_generator(train_generator_open, epochs=15, verbose=1,validation_data=test_generator_open)#,callbacks=[checkpointer])
        prediction = model.predict_generator(test_generator_open)
        
        def predicts(num_prediction, model):
            prediction_list = close_data[-look_back:]

            for _ in range(num_prediction):

                y = prediction_list[-look_back:]
                y = y.reshape((1, look_back, 1))
                inn = model.predict(y)[0][0]
                prediction_list = np.append(prediction_list, inn)
            prediction_list = prediction_list[look_back-1:]

            return prediction_list

        num_prediction = 15
        forecast_close = predicts(num_prediction, model).tolist()
        return jsonify(forecast_close)
    # except:
    #     return jsonify("Please Pass correct excel/csv file.")
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)