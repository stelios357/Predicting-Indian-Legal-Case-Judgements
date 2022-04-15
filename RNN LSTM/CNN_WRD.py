#from glob import glob
#import os
import math
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import tensorflow as tf
from tensorflow import keras
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import plot_model

#look_back = 210

class Dataset:
    def __init__(self,look_back=120, trainX=0, trainY=0, testX=0, testY=0, validX=0, validY=0, trainLabels=0, testLabels=0, validLabels=0 ):
       self.look_back = look_back

       self.trainX = np.empty(shape=(0, self.look_back+2), dtype=object)
       self.trainY = np.empty(shape=(0), dtype=object)
       self.trainLabels = np.empty(shape=(0), dtype=object)

       self.testX = np.empty(shape=(0, self.look_back+2), dtype=object)
       self.testY = np.empty(shape=(0), dtype=object)
       self.testLabels = np.empty(shape=(0), dtype=object)

       self.validX = np.empty(shape=(0, self.look_back+2), dtype=object)
       self.validY = np.empty(shape=(0), dtype=object)
       self.validLabels = np.empty(shape=(0), dtype=object)

    def __add__(self, other):

        self.trainX = np.float16(np.vstack((self.trainX, other.trainX)))
        self.trainY = np.float16(np.hstack((self.trainY, other.trainY)))
        self.trainLabels = np.float16(np.hstack((self.trainLabels, other.trainLabels)))

        self.validX = np.float16(np.vstack((self.validX, other.validX)))
        self.validY = np.float16(np.hstack((self.validY, other.validY)))
        self.validLabels = np.float16(np.hstack((self.validLabels, other.validLabels)))

        self.testX  = np.float16(np.vstack((self.testX, other.testX)))
        self.testY  = np.float16(np.hstack((self.testY, other.testY)))
        self.testLabels  = np.float16(np.hstack((self.testLabels, other.testLabels)))
        return self



def readAndPreprocessData():
    df = pd.read_csv('WRD_RSMR_Dataset.csv')

    df = df.drop(columns=['sno','catchment_code','catchment_name','district_name'])
    df['year'] = df['year'].astype('uint16')
    df['latitude'] = df['latitude'].astype('uint16')
    df['longitude'] = df['longitude'].astype('uint16')
    df['coordinates'] = df['coordinates'].astype('uint64')
    df['coordinates'] = df['coordinates'].astype('str')
#    df.info(verbose=False, memory_usage="deep")

    column_names = ['year', 'latitude', 'longitude', 'coordinates', 'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

    float_column_name = column_names[4:]
    for i in range(len(float_column_name)):
        df[float_column_name[i]] = df[float_column_name[i]].astype('float16')

#    df['sno'] = df['sno'].astype('uint16')

#    df.info(verbose=False, memory_usage="deep")
    print(df)
    print(df.shape)

#    tt=input()

    return df

def create_dataset(data, dataLabels, look_back=1, latitude1=0, longitude1=0):
    dataX, dataY, labelY = [], [], []
#    print("in create_dataset")
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
#        print(a)
#        print(a.shape)
        a = np.insert(a,0,latitude1/100)
        a = np.insert(a,1,longitude1/100)
#        print(a)
#        a = [latitude1] + [longitude1] + a
        dataX.append(a)
#        print(dataX)
#        tt = input()
        dataY.append(data[i + look_back, 0])
        labelY.append(dataLabels[i + look_back, 0])
#        print(dataX)
#        print("______________________________")
#        print(dataY)
#        print("______________________________")
#        print(labelY)
#        tt=input()
#    return np.array(dataX,dtype=np.float16), np.array(dataY,dtype=np.float16)
  
    return np.array(dataX), np.array(dataY), np.array(labelY)


def convertCoordinateDataIntoList(coord_df, startYear, endYear):

    dataList = list()
    for year in range(startYear, endYear):
        tempList = coord_df.loc[coord_df['year'] == year]
        tempList = tempList.values.tolist()
        tempList = tempList[0][4:]
        dataList = dataList + tempList

#    print(dataList)
#    tt = input()
    return dataList


def splitData(coordinates, df, look_back, start_Year, SplitYearTrain, SplitYearValid, end_Year, dataset):
    for coords in coordinates:

        print(coords)
#        print("%%%%%%%%%%%")
#        tt=input()
        df_coords = df[df['coordinates'] == coords]
    
        trainData = convertCoordinateDataIntoList(df_coords, start_Year, SplitYearTrain)
        validData =  convertCoordinateDataIntoList(df_coords, SplitYearTrain, SplitYearValid)
        testData =  convertCoordinateDataIntoList(df_coords, SplitYearValid, end_Year)
#        validData = convertCoordinateDataIntoList(df_coords, SplitYearValid, end_Year)
        latitude = int(str(coords)[:4])
        longitude = int(str(coords)[4:])
    #    print("Latitude, Longitude ", latitude, longitude)
    #    print("Latitude, Longitude ", latitude, longitude)
#        print(len(trainData))
#        print(len(testData))
#        print(len(validData))
#        print(validData)
#        tt = input()
        ## To identify the months for which predictions are made,
        ## we are creating a separate list for month numbers
        month_labels = [1,2,3,4,5,6,7,8,9,10,11,12]
        month_labels_train = month_labels*int((len(trainData)/12))
        month_labels_test = month_labels*int((len(testData)/12))
        month_labels_valid = month_labels*int((len(validData)/12))
#        print(len(month_labels_train))
#        print(len(month_labels_test))
#        tt = input()

        trainData = np.reshape(trainData,(-1,1))
        validData = np.reshape(validData,(-1,1))
        testData = np.reshape(testData,(-1,1))

        month_labels_train = np.reshape(month_labels_train,(-1,1))
        month_labels_test = np.reshape(month_labels_test,(-1,1))
        month_labels_valid = np.reshape(month_labels_valid,(-1,1))

        #Normalize the data in range 0-300
    #     scaler = MinMaxScaler(feature_range=(0,300))
    #     scaler1 = MinMaxScaler(feature_range=(0,300))
    #     trainData = scaler.fit_transform(trainData)
    #     testData = scaler1.fit_transform(testData)
    #    print(trainData)
#        print(vars(dataset))

        td = Dataset()
#        print("^^^^^^")
#        print(vars(td))
        td.trainX, td.trainY, td.trainLabels = create_dataset(trainData, month_labels_train, look_back, latitude, longitude)
        td.validX, td.validY, td.validLabels = create_dataset(validData, month_labels_valid, look_back, latitude, longitude)
        td.testX, td.testY, td.testLabels = create_dataset(testData, month_labels_test, look_back, latitude, longitude)
        
#        print(len(td.testY))
#        print(td.testY.shape)
#        print("-----test labels months----")
#        print(td.testLabels.shape)
#        print(len(td.trainY))
#        print(td.trainY.shape)
#        print("-----train labels months----")
#        print(td.trainLabels.shape)
#        print(type(td.trainLabels))
#        print(type(td.trainY))
#        tt=input()
    
#        print(td.trainX)
#        print(td.validX)
#
#        print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
#
#        print(td.trainY)
#
#        tt = input()
#        print("-------------------------------") 
#        print(type(dataset))
#        print(type(dataset.trainX))
#        print(dataset.trainX.shape)
#        print(type(td))
        dataset += td
#        dataset.trainlabels = np.float16(np.hstack((td.trainLabels, dataset.trainLabels)))
#        print(dataset.trainX.shape)
#        print(dataset.trainY.shape)
#        print(type(dataset.trainY))
#        print(type(dataset.trainLabels))
#        print(dataset.trainLabels.shape)
#        print(dataset.testLabels.shape)
#        print(dataset.validLabels.shape)
#        print(dataset.trainY)
#        print(dataset.trainLabels)
#        print("__________dataset__________")

    
def normalizeDataset(data, dataR, minValue=0, maxValue=100):

    scaler = MinMaxScaler(feature_range=(minValue, maxValue))
    dataR.trainX = scaler.fit_transform(data.trainX[:,2:])
    dataR.validX = scaler.fit_transform(data.validX[:,2:])
    dataR.testX = scaler.fit_transform(data.testX[:,2:])

    data.trainY = data.trainY.reshape(-1,1)
    dataR.trainY = scaler.fit_transform(data.trainY)

    data.validY = data.validY.reshape(-1,1)
    dataR.validY = scaler.fit_transform(data.validY)

    data.testY = data.testY.reshape(-1,1)
    dataR.testY = scaler.fit_transform(data.testY)



def replaceNegativeValues(d):
    d.trainX[d.trainX<0] = 0
    d.testX[d.testX<0] = 0
    d.validX[d.validX<0] = 0
    d.trainY[d.trainY<0] = 0
    d.testY[d.testY<0] = 0
    d.validY[d.validY<0] = 0

def getCNN(dataR_shape, n_filters):
    
    i1 = keras.layers.Input(shape = dataR_shape[1])
    i1_reshape = keras.layers.Reshape((dataR_shape[1], 1))(i1)
    i2 = keras.layers.Input(shape=[2])
    c1 = keras.layers.Conv1D(filters=n_filters, kernel_size=5, strides=1, padding= 'causal', activation='relu', input_shape=(dataR_shape[1],1))(i1_reshape)
    c2 = keras.layers.Conv1D(filters=n_filters, kernel_size=5, strides=1, padding='causal', activation='relu')(c1)
    d1 = keras.layers.Dropout(rate=0.2)(c2)
    p1 = keras.layers.GlobalAveragePooling1D()(d1)
    concat = keras.layers.Concatenate()([i2,p1])
    output = keras.layers.Dense(1)(concat)

    return keras.models.Model(inputs=[i1, i2], outputs = [output])

def compileAndFitModel(dataR, dataC, model):
    
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=['mae','mse'])
    checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath="bestCNN_WRD.h5", monitor='val_loss', save_best_only=True)
    history = model.fit((dataR.trainX, dataC.trainX), dataR.trainY, epochs=200, validation_data=((dataR.validX, dataC.validX), dataR.validY), callbacks=[checkpoint_cb], batch_size=32, verbose=2)
    return keras.models.load_model("bestCNN_WRD.h5")


    

def main():
    df = readAndPreprocessData()
    coordinates = np.unique(df['coordinates'])
#    coordinates = coordinates[:2]

    latitudes = np.unique(df['latitude'])
    longitudes = np.unique(df['longitude'])

    print(coordinates)
    print(len(coordinates))
    print(df)
    #tt = input()
    #SplitYear = 2007
    start_Year = 1957#1901#1957
    SplitYearTrain = 1996#1991#2007
    SplitYearValid = 2007#2006#2015
    end_Year = 2018#2018
    look_back = 108#210

    dataset = Dataset(look_back)

    splitData(coordinates, df, look_back, start_Year, SplitYearTrain, SplitYearValid, end_Year, dataset)
    print(dataset.trainX)
    print(dataset.trainY)
#    print(type(trainX))
#    print(type(trainY))
    print(dataset.trainX.shape)
    print(dataset.trainY.shape)

    # Datatset contains -1 in place of missing values
    # Make it 0, otherwise negative values can adversely affect the learning

    replaceNegativeValues(dataset)

    # Latitude and longitude are in range 0-100 while rainfall values
    # are from 0-more that 500
    # Normalize only the rainfall values in the range of 0-100

    dataCoords = Dataset(0)
    dataCoords.trainX = dataset.trainX[:,:2]
    dataCoords.validX = dataset.validX[:,:2]
    dataCoords.testX = dataset.testX[:,:2]

    dataRainfall = Dataset(look_back-2)

    scaler = MinMaxScaler(feature_range=(0, 100))
    dataRainfall.trainX = scaler.fit_transform(dataset.trainX[:,2:])
    dataRainfall.validX = scaler.fit_transform(dataset.validX[:,2:])
    dataRainfall.testX = scaler.fit_transform(dataset.testX[:,2:])

    dataset.trainY = dataset.trainY.reshape(-1,1)
    dataRainfall.trainY = scaler.fit_transform(dataset.trainY)

    dataset.validY = dataset.validY.reshape(-1,1)
    dataRainfall.validY = scaler.fit_transform(dataset.validY)

    dataset.testY = dataset.testY.reshape(-1,1)
    dataRainfall.testY = scaler.fit_transform(dataset.testY)


#    normalizeDataset(dataset, dataRainfall, 0, 100)
    print(np.amin(dataRainfall.trainX))
    print(np.amax(dataRainfall.trainX))

    print(np.amin(dataRainfall.trainY))
    print(np.amax(dataRainfall.trainY))


    keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)

    dataRTrainXShape = dataRainfall.trainX.shape
    model = getCNN(dataRTrainXShape, n_filters=100)

    print(model.summary())
#    plot_model(model, to_file='./CNN.png')

#    tt = input()
    
    model = compileAndFitModel(dataRainfall, dataCoords, model)

    trainPredict = model.predict((dataRainfall.trainX, dataCoords.trainX))
    validPredict = model.predict((dataRainfall.validX, dataCoords.validX))
    testPredict = model.predict((dataRainfall.testX, dataCoords.testX))

    # Prediction before scaling of data
    print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~")
    print("Before scaling of data")
    trainScore = keras.metrics.mean_absolute_error(dataRainfall.trainY[:,0],trainPredict[:,0]).numpy()
    validScore = keras.metrics.mean_absolute_error(dataRainfall.validY[:,0],validPredict[:,0]).numpy()
    testScore = keras.metrics.mean_absolute_error(dataRainfall.testY[:,0],testPredict[:,0]).numpy()
    print( 'Train Score: %.4f MAE' % (trainScore))
    print( 'Valid Score: %.4f MAE' % (validScore))
    print( 'Test Score: %.4f MAE' % (testScore))
    
    trainScore = math.sqrt(mean_squared_error(dataRainfall.trainY[:,0], trainPredict[:,0]))
    print( 'Train Score: %.4f RMSE' % (trainScore))
    validScore = math.sqrt(mean_squared_error(dataRainfall.validY[:,0], validPredict[:,0]))
    print( 'Valid Score: %.4f RMSE' % (validScore))
    testScore = math.sqrt(mean_squared_error(dataRainfall.testY[:,0], testPredict[:,0]))
    print( 'Test Score: %.4f RMSE' % (testScore))
    
    print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~")

    plt.plot(dataRainfall.testY[-360:])
    plt.plot(testPredict[-360:])
    # plt.plot(testPredict[-366:])
    plt.savefig("plt_unNorm.png")
    plt.show()

    #invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    dataRainfall.trainY = scaler.inverse_transform(dataRainfall.trainY)
    testPredict = scaler.inverse_transform(testPredict)
    dataRainfall.testY = scaler.inverse_transform(dataRainfall.testY)

    print("After scaling of data")
    trainScore = keras.metrics.mean_absolute_error(dataRainfall.trainY[:,0],trainPredict[:,0]).numpy()
    testScore = keras.metrics.mean_absolute_error(dataRainfall.testY[:,0],testPredict[:,0]).numpy()
    '''
    trainScore = keras.metrics.mean_absolute_error(trainY,trainPredict[:,0]).numpy()
    testScore = keras.metrics.mean_absolute_error(testY,testPredict[:,0]).numpy()
    '''
    print( 'Train Score: %.2f MAE' % (trainScore))
    print( 'Test Score: %.2f MAE' % (testScore))
    
    
    trainScore = math.sqrt(mean_squared_error(dataRainfall.trainY[:,0], trainPredict[:,0]))
    print( 'Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(dataRainfall.testY[:,0], testPredict[:,0]))
    print( 'Test Score: %.2f RMSE' % (testScore))


if __name__ == '__main__':
    main()


