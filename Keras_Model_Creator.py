import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.backend import sigmoid
from keras.activations import tanh
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation


# Creating the swish activation function
# Never used, because of its dependencies.
# (It needs to be re-introduced to even use a model with swish in it, so
# that's a no-no).
def swish(x, beta=1):
    return (x * sigmoid(beta * x))


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
# or lower than the real stuff, we sway the chart to look more volatile, it
# won't change the error margin that much, but it should get us more rigid,
# more chiseled data, instead of the smoothed-out by the books BS.
def volatilityDistortion(rawY, volatilityScaler=2):
    reworkedPrices = np.copy(rawY)
    for i in range(1, rawY.shape[0]):
        reworkedPrices[i] += (rawY[i]-rawY[i-1])*(
            1+volatilityScaler)*(2*abs(rawY[i]-rawY[i-1])/(rawY[i]+rawY[i-1]))

    # Test plotting to tweak and tune the distortion algorithm
    plt.plot(rawY, 'green')
    plt.plot(reworkedPrices, 'orange')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Stock Price Distortion')
    plt.show()

    return np.array(reworkedPrices)


# PREPROCESSING
def createModel(name='MSFT', look_back=7):

    # Allows us to use swish as an activation function
    get_custom_objects().update({'swish': Activation(swish)})
    datasetRaw = pd.read_csv("data/" + name + ".csv", usecols=['Close'])
    datasetRaw = datasetRaw.values

    datasetRaw = datasetRaw.astype('float32')

    # Always scale your data, Kev my man!
    scaler = MinMaxScaler(feature_range=(0, 1))
    datasetRaw = np.reshape(datasetRaw, (-1, 1))
    datasetRaw = scaler.fit_transform(datasetRaw)
    totalX, totalY = createDataset(datasetRaw, look_back)

    # Scaling the distorted data
    # datasetDistorted = volatilityDistortion(datasetRaw[look_back:])
    # datasetDistorted = np.reshape(datasetDistorted, (-1, 1))
    # datasetDistorted = scaler.fit_transform(datasetDistorted)

    # Changing the training "correct" data to the distorted,
    # more volatile dataset
    # totalY = datasetDistorted
    totalY = volatilityDistortion(totalY)
    print(totalX.shape)
    print(totalY.shape)

    # Splitting the data randomly
    trainX, testX, trainY, testY = train_test_split(
        totalX, totalY, test_size=0.25)

    # Reformatting the training and testing data
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Creating the model actual model
    # IDK Dense layers just don't work my man, what can I tell you LMAO
    # Took all the Dense layers away, decided to hone this to a
    # strictly LSTM model more or less, so there's that.
    model = Sequential()
    model.add(LSTM(96, input_shape=(
        1, look_back), return_sequences=True, activation='relu'))
    model.add(LSTM(96, input_shape=(
        1, look_back), return_sequences=True, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(0.01))
    model.add(LSTM(96, input_shape=(
        1, look_back), return_sequences=True, activation='relu'))
    model.add(LSTM(96, input_shape=(
        1, look_back), return_sequences=True, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    model.fit(trainX, trainY, epochs=1024, batch_size=24, verbose=1)

    # Saving the model
    model.save('lstm_model_'+name+'.h5')
    # executeModel(totalX, model, look_back)


createModel('MSFT', 7)
