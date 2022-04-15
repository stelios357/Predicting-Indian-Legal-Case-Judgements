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


def create_dataset(data, look_back=1, latitude1=0, longitude1=0):
    dataX, dataY = [], []
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
#    return np.array(dataX,dtype=np.float16), np.array(dataY,dtype=np.float16)
  
    return np.array(dataX), np.array(dataY)


#dropCols = ['sno','catchment_code','catchment_name','district_name','latitude','longitude','misc1','misc2','jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
dropCols = ['sno','catchment_code','catchment_name','district_name','misc1','misc2','jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']


df = pd.read_csv('rainfallDataset2.csv')
#print(df)

df = df[~df.catchment_name.str.contains("IMD")]
matchstr = r'\(A'
df = df[~df['catchment_name'].str.contains(matchstr)]
df = df.drop(columns=dropCols)
print(df)
for i in range(1,367):
    df[str(i)] = df[str(i)].astype('float16')

df['year'] = df['year'].astype('uint16')
df['latitude'] = df['latitude'].astype('uint16')
df['longitude'] = df['longitude'].astype('uint16')

#df['coordinates'] = df['coordinates'].astype('uint32')
#print(df.dtypes)
#print(df)
df.info(verbose=False, memory_usage="deep")

#tt = input()

#index_names = df['(' in df['catchment_name']].index
#print(index_names)

#df.drop(df['(' in df['catchment_name']].index, inplace=True)
print(df.shape)

coordinates = np.unique(df['coordinates'])
#coordinates = coordinates[:2]

latitudes = np.unique(df['latitude'])
longitudes = np.unique(df['longitude'])

print(coordinates)
print(len(coordinates))
print(df)
#tt = input()
#SplitYear = 2007
SplitYearTrain = 2007
SplitYearValid = 2015
look_back = 210#240#200#366
#dataset = np.empty(366*50)
#dataset = []
# trainX = [None]*look_back
# trainY = [None]
trainX = np.empty(shape=(0,look_back+2), dtype=object)
trainY = np.empty(shape=(0),dtype=object)
testX = np.empty(shape=(0,look_back+2), dtype=object)
testY = np.empty(shape=(0),dtype=object)
validX = np.empty(shape=(0,look_back+2), dtype=object)
validY = np.empty(shape=(0),dtype=object)


for coords in coordinates:
    df_coords = df[df['coordinates'] == coords]
#    print(df_coords)
#    tt=input()

#    df_coords.sort_values(by = ['year'], ascending=[True])
    trainData = list()
    testData = list()
    validData = list()
    latitude = 0
    longitude = 0
    for year in range (1957, SplitYearTrain):
        tempList = df_coords.loc[df_coords['year'] == year]
        tempList = tempList.values.tolist()
#        print(tempList)
#        tt = input()

#        tempList = tempList[0][2:368]
        latitude = tempList[0][1]
        longitude = tempList[0][2]
        tempList = tempList[0][4:370]
#        print(len(tempList))
#        tt = input()
#         print(type(tempList))
#         print(len(tempList))
        trainData = trainData + tempList

    for year in range (SplitYearTrain, SplitYearValid):
        tempList = df_coords.loc[df_coords['year'] == year]
        tempList = tempList.values.tolist()
        #print(tempList)
        tempList = tempList[0][4:370]
#        print(len(tempList))
#        tt = input()
#         print(type(tempList))
#         print(len(tempList))
        validData = validData + tempList

# 
#         print(type(trainData))
#         print(len(trainData))
   
    for year in range (SplitYearValid, 2018):
        tempList = df_coords.loc[df_coords['year'] == year]
        tempList = tempList.values.tolist()
        tempList = tempList[0][4:370]
        testData = testData + tempList

    
    trainData = np.reshape(trainData,(-1,1))
    validData = np.reshape(validData,(-1,1))
    testData = np.reshape(testData,(-1,1))
    #Normalize the data in range 0-300
#     scaler = MinMaxScaler(feature_range=(0,300))
#     scaler1 = MinMaxScaler(feature_range=(0,300))
#     trainData = scaler.fit_transform(trainData)
#     testData = scaler1.fit_transform(testData)
#    print(trainData)
    trainDatasetX, trainDatasetY = create_dataset(trainData, look_back, latitude, longitude)
    validDatasetX, validDatasetY = create_dataset(validData, look_back, latitude, longitude)
    testDatasetX, testDatasetY = create_dataset(testData, look_back, latitude, longitude)
#     print(datasetX)
#    print(type(datasetX))
#     print(datasetX.shape)
#     print(datasetY)
#     print(type(datasetY))
#     tt = input()

#     print(datasetY.shape)
#    np.concatenate(trainX, datasetX)
    trainX = np.vstack((trainX, trainDatasetX))
    trainX = np.float16(trainX)
#    print(round(sys.getsizeof(trainX)/1024/1024,2))
    print(trainX)
#    tt = input()
#    np.concatenate(trainY, datasetY)
    trainY = np.hstack((trainY, trainDatasetY))
    trainY = np.float16(trainY)

    validX = np.vstack((validX, validDatasetX))
    validX = np.float16(validX)
    validY = np.hstack((validY, validDatasetY))
    validY = np.float16(validY)


    testX = np.vstack((testX, testDatasetX))
    testX = np.float16(testX)
    testY = np.hstack((testY, testDatasetY))
    testY = np.float16(testY)

    print(round(sys.getsizeof(trainX)/1024/1024,2))
    
#     print(trainX)
#     print(trainY)
#    print(type(trainX))
#    print(type(trainY))
    print(trainX.shape)
    print(trainY.shape)
#     print(trainX)
#     print(trainY)
#    t = input()

print("******training and test set preparation Done *******")
df=0
df_coords = 0
trainData = 0
testData = 0
trainDatasetX=0
trainDatasetY = 0
testDatasetX=0
testDatasetY=0
validData = 0
validDatasetX=0
validDatasetY=0
print(np.amin(trainX))

trainX[trainX<0] = 0
testX[testX<0] = 0
validX[validX<0] = 0

trainY[trainY<0] = 0
testY[testY<0] = 0
validY[validY<0] = 0

# print(np.amin(trainX))
# print(np.amax(trainX))
# 

# scaler = MinMaxScaler(feature_range=(0,300))
# scaler1 = MinMaxScaler(feature_range=(0,300))
# trainX = scaler.fit_transform(trainX)
# trainY = scaler.fit_transform([trainY])
# testY = scaler1.fit_transform([testY])
# 
#normalize the dataset
print(trainY.shape)
scaler = MinMaxScaler(feature_range=(0,100))
trainX1 = scaler.fit_transform(trainX[:,2:])
#trainX2 = scaler.fit_transform(trainX[:,:2])
trainX2 = trainX[:,:2]
#trainX = scaler.fit_transform(trainX)
#np.reshape(trainY,(-1,1))
trainY = trainY.reshape(-1,1)

trainY = scaler.fit_transform(trainY)

testX1 = scaler.fit_transform(testX[:,2:])
#testX2 = scaler.fit_transform(testX[:,:2])
testX2 = testX[:,:2]
#testX = scaler.fit_transform(testX)

testY = testY.reshape(-1,1)
testY = scaler.fit_transform(testY)

validX1 = scaler.fit_transform(validX[:,2:])
#validX2 = scaler.fit_transform(validX[:,:2])
validX2 = validX[:,:2]
#validX = scaler.fit_transform(validX)

validY = validY.reshape(-1,1)
validY = scaler.fit_transform(validY)


print(np.amin(trainX))
print(np.amax(trainX))
print(np.amin(trainY))
print(np.amax(trainY))

#t = input()

print(np.amin(testX))
print(np.amax(testX))
print(testY.shape)
print(testX.shape)

#t = input()

keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

#trainX = trainX.reshape(trainX.shape[0],trainX.shape[1],1)

#input_shape = (samples, n_timesteps, n_features)
#inputLayer = keras.layers.Input(shape = trainX.shape[1:])


# following model was giving best results so far
'''
inputLayer = keras.layers.Input(shape = trainX.shape[1])
inputLayer1 = keras.layers.Reshape((trainX.shape[1],1))(inputLayer)
drop1 = keras.layers.Dropout(rate=0.2)(inputLayer)
hidden1 = keras.layers.Dense(300,activation="relu",kernel_initializer='he_normal')(drop1)
drop2 = keras.layers.Dropout(rate=0.2)(hidden1)
hidden2 = keras.layers.Dense(100,activation="relu",kernel_initializer='he_normal')(drop2)
drop3 = keras.layers.Dropout(rate=0.2)(hidden2)
hidden3 = keras.layers.Dense(50,activation="relu",kernel_initializer='he_normal')(drop3)
#convLayer = keras.layers.Conv1D(filters=50, kernel_size=5, strides=1, activation='relu', input_shape=(trainX.shape[1],1))(inputLayer1)
convLayer = keras.layers.Conv1D(filters=50, kernel_size=5, strides=1, padding="causal", activation='relu', input_shape=(trainX.shape[1],1))(inputLayer1)
#flattenLayer = keras.layers.Flatten()(convLayer)
flattenLayer = keras.layers.GlobalAveragePooling1D()(convLayer)
concat = keras.layers.Concatenate()([flattenLayer, hidden3])

output1 = keras.layers.Dense(1)(concat)
output2 = keras.layers.Lambda(lambda x: x*300)(output1)
model = keras.models.Model(inputs=[inputLayer], outputs = [output2])
'''

#inputLayer = keras.layers.Input(shape = trainX.shape[1])
inputLayer = keras.layers.Input(shape = trainX1.shape[1])
inputLayer1 = keras.layers.Reshape((trainX1.shape[1],1))(inputLayer)
inputLayer2 = keras.layers.Input(shape=[2])
convLayer1 = keras.layers.Conv1D(filters=100, kernel_size=5, strides=1, padding="causal", activation='relu', input_shape=(trainX1.shape[1],1))(inputLayer1)
convLayer2 = keras.layers.Conv1D(filters=100, kernel_size=5, strides=1, padding="causal", activation='relu')(convLayer1)
drop1 = keras.layers.Dropout(rate=0.2)(convLayer2)
#pooling1 = keras.layers.MaxPooling1D(3)(drop1)
#convLayer3 = keras.layers.Conv1D(filters=100, kernel_size=5, strides=1, padding="causal", activation='relu')(pooling1)
#pooling2 = keras.layers.GlobalAveragePooling1D()(convLayer3)
pooling2 = keras.layers.GlobalAveragePooling1D()(drop1)
#drop2 = keras.layers.Dropout(rate=0.2)(pooling2)
#concat = keras.layers.Concatenate()([drop2, inputLayer2])
concat = keras.layers.Concatenate()([pooling2, inputLayer2])

output1 = keras.layers.Dense(1)(concat)
#output2 = keras.layers.Dense(1)(output1)
#output2 = keras.layers.Lambda(lambda x: x*300)(output1)
model = keras.models.Model(inputs=[inputLayer, inputLayer2], outputs = [output1])


print(model.summary())
plot_model(model, to_file='./CNN.png')


#t = input()
#model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(), metrics=["mae"])

model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=['mae','mse'])

#without validation data
'''
history = model.fit(trainX, trainY,epochs=50,batch_size=16,verbose=2)
model.save("myModel.h5")
'''
# check point to save best model
#checkpoint_cb = keras.callbacks.ModelCheckpoint("modelDWNN.h5", save_best_only=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath="bestCNN.h5", monitor='val_loss', save_best_only=True)

history = model.fit((trainX1,trainX2), trainY,epochs=50, validation_data = ((validX1, validX2), validY),callbacks=[checkpoint_cb], batch_size=32,verbose=2)


history_dict = history.history
print(history_dict.keys())
# t = input()
# plt.semilogx(history.history['lr'], history.history['loss'])
# plt.axix([1e-8,1e-3,0,300])
# plt.show()

# generate predictions for training

model = keras.models.load_model("bestCNN.h5")

'''
trainPredict = model.predict((trainX[:,2:],trainX[:,:2]))
validPredict = model.predict((validX[:,2:],validX[:,:2]))
testPredict = model.predict((testX[:,2:],testX[:,:2]))
'''
trainPredict = model.predict((trainX1,trainX2))
validPredict = model.predict((validX1,validX2))
testPredict = model.predict((testX1,testX2))


# Prediction before scaling of data
print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~")
print("Before scaling of data")
trainScore = keras.metrics.mean_absolute_error(trainY[:,0],trainPredict[:,0]).numpy()
validScore = keras.metrics.mean_absolute_error(validY[:,0],validPredict[:,0]).numpy()
testScore = keras.metrics.mean_absolute_error(testY[:,0],testPredict[:,0]).numpy()
print( 'Train Score: %.4f MAE' % (trainScore))
print( 'Valid Score: %.4f MAE' % (validScore))
print( 'Test Score: %.4f MAE' % (testScore))

trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print( 'Train Score: %.4f RMSE' % (trainScore))
validScore = math.sqrt(mean_squared_error(validY[:,0], validPredict[:,0]))
print( 'Valid Score: %.4f RMSE' % (validScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print( 'Test Score: %.4f RMSE' % (testScore))

print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~")

'''
plt.plot(history.history['mse'])
plt.show()
'''
'''
#print(trainPredict)
print(trainPredict.shape)
print(trainY.shape)
print(testPredict.shape)
'''
print("****************************************************")

'''
print(np.amin(testY))
print(np.amax(testY))
print(np.amin(testPredict))
print(np.amax(testPredict))
'''


#invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
# 
print('\n~*~*~*~*~*~~~~~~~*~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(np.amin(testY))
print(np.amax(testY))
print(np.amin(testPredict))
print(np.amax(testPredict))

# 
# # plt.plot(scaler.inverse_transform(dataset[-366:]))
# plt.show()
#plt.plot(scaler.inverse_transform(testY[-366:]))
# plt.plot(testPredict[-366:])
# plt.show()
plt.plot(testY[-366:])
plt.plot(testPredict[-366:])
# plt.plot(testPredict[-366:])
plt.show()

# plt.plot(trainPredict[-366:])
# plt.show()
plt.plot(trainY[-366:])
plt.plot(trainPredict[-366:])
# plt.plot(testPredict[-366:])
plt.show()


#print(trainPredict.shape)

trainMax = np.amax(testX)
#trainY = np.reshape(trainY,(-1,1))
#testY = np.reshape(testY,(-1,1))
#trainY = np.reshape(trainY,(-1,1))
#trainY = np.reshape(trainY,(-1,1))
scaler = MinMaxScaler(feature_range=(0,200))
#scaler1 = MinMaxScaler(feature_range=(0,300))
trainY = scaler.fit_transform(trainY)
trainPredict = scaler.fit_transform(trainPredict)
testY = scaler.fit_transform(testY)
testPredict = scaler.fit_transform(testPredict)


plt.plot(testPredict[-1098:])
plt.show()
plt.plot(testY[-1098:])
plt.plot(testPredict[-1098:])
# plt.plot(testPredict[-366:])
plt.show()

plt.plot(trainPredict[-1098:])
plt.show()
plt.plot(trainY[-1098:])
plt.plot(trainPredict[-1098:])
# plt.plot(testPredict[-366:])
plt.show()

# plt.plot(testPredict[-300:-100])
# plt.show()
plt.plot(testY[-300:-100])
plt.plot(testPredict[-300:-100])
# plt.plot(testPredict[-366:])
plt.show()

# plt.plot(trainPredict[-300:-100])
# plt.show()
plt.plot(trainY[-300:-100])
plt.plot(trainPredict[-300:-100])
# plt.plot(testPredict[-366:])
plt.show()



#print(trainPredict.shape)
'''
print(trainY[:,0])
print(trainPredict[:,0])
'''
print("After scaling of data")
trainScore = keras.metrics.mean_absolute_error(trainY[:,0],trainPredict[:,0]).numpy()
testScore = keras.metrics.mean_absolute_error(testY[:,0],testPredict[:,0]).numpy()
'''
trainScore = keras.metrics.mean_absolute_error(trainY,trainPredict[:,0]).numpy()
testScore = keras.metrics.mean_absolute_error(testY,testPredict[:,0]).numpy()
'''
print( 'Train Score: %.2f MAE' % (trainScore))
print( 'Test Score: %.2f MAE' % (testScore))


trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print( 'Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print( 'Test Score: %.2f RMSE' % (testScore))

'''
print(trainY)
print(trainPredict[:,0])
trainScore = keras.metrics.mean_absolute_error(trainY,trainPredict[:,0]).numpy()
testScore = keras.metrics.mean_absolute_error(testY,testPredict[:,0]).numpy()
print( 'Train Score: %.2f MAE' % (trainScore))
print( 'Test Score: %.2f MAE' % (testScore))


trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print( 'Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print( 'Test Score: %.2f RMSE' % (testScore))
'''


