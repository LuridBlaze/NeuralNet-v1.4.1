import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras.backend import sigmoid


# Creates sequences of look_back long intervals from a dataframe
# To be exact, it feeds it the stock, + the length, then returns:
# [[1, 2, .., n = look_back], ... , len-look_back] + [look_back, ..., len-1]
# So basically a very well done division of the train or test data.
def createDataset(dataset, look_back=7):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        temp = dataset[i:(i+look_back), 0]
        dataX.append(temp)
        dataY.append([dataset[i+look_back, 0]])

    return np.array(dataX), np.array(dataY)


# An ingenious Idea to distort the data, the reason this works better is:
# since it's important for us whether or not the predicted value falls higher
# or lower than the real stuff^TM, we sway the chart to look more volatile, it
# won't change the error margin that much, but it should get us more rigid,
# more chiseled data, instead of the smoothed-out by the books BS.
def volatilityDistortion(rawY, volatilityScaler=2):
    reworkedPrices = np.copy(rawY)
    for i in range(1, rawY.shape[-1]):
        reworkedPrices[i] += (rawY[i]-rawY[i-1])*(
            1+volatilityScaler)*(2*abs(rawY[i]-rawY[i-1])/(rawY[i]+rawY[i-1]))

    return np.array(reworkedPrices)


# Launching the model
def executeModel(name='MSFT', model=load_model('data/lstm_model_MSFT.h5'),
                 look_back=7, predictLastUnits=1000):
    # importlib.import_module('Behavioural_Keras_Model_h5')
    # Loading up the data
    datasetRaw = pd.read_csv("data/" + name + ".csv", usecols=['Close'])
    datasetRaw = datasetRaw.values
    # datasetRaw = volatilityDistortion(datasetRaw)
    datasetRaw = datasetRaw.astype('float32')

    # SCALING THE INPUT
    scaler = MinMaxScaler(feature_range=(0, 1))
    datasetRaw = np.reshape(datasetRaw, (-1, 1))
    datasetRaw = scaler.fit_transform(datasetRaw)
    data, y = createDataset(datasetRaw, look_back)
    y = volatilityDistortion(y)
    predictedY = np.empty_like(datasetRaw)
    predictedY[::] = np.nan
    print(data.shape, 'X')
    print(y.shape, 'Y')
    print(datasetRaw.shape, 'X + Y')
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))

    # Flattening out the correct stock data
    # predictLastUnits = len(data)
    for i in range(predictLastUnits):
        new_data = data[-predictLastUnits+i]
        print(new_data)
        new_data = np.reshape(new_data, (1, 1, look_back))
        prediction = model.predict(new_data)
        # print(prediction)
        prediction = scaler.inverse_transform(prediction[0])
        predictedY[-predictLastUnits+i-1] = prediction
    datasetRaw = scaler.inverse_transform(datasetRaw)
    datasetRaw = np.reshape(datasetRaw, (datasetRaw.shape[0]))
    predictedY = np.reshape(predictedY, (predictedY.shape[0]))
    # Displaying the predicted data, layered on top of the actual and the
    # volatile dataset we created along the way,
    # from which we take the training really.
    plt.plot(datasetRaw, 'green')
    plt.plot(y, 'orange')
    plt.plot(predictedY, 'red')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    print(datasetRaw.shape, 'X')
    print(predictedY.shape, 'Y')
    plt.show()


executeModel('MSFT', load_model('data/lstm_model_MSFT.h5'), 7)
