
#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error


# Creating  a function to predict the stock price for selected stocks using
def predictPriceLR(stockname):
    name=stockname.split('_')
    print('Stock Analyzed:',name[0])
    data_array=pd.read_excel(stockname)
    data_array=data_array.dropna()
    data_array['Price_Difference']=data_array['Last Price']-data_array['Closing Price 1 Day Ago']
    data_array['Price Range']=data_array['High Price']-data_array['Low Price']
    #sorting by date
    data_array=data_array.sort_values(by=['Date'])
    #getting train and test data
    data_array=data_array.sort_values(by=['Date'])
    xtrain=data_array.iloc[:-30,[1,6,7,8,9,12,13]]
    ytrain=data_array.iloc[1:-29, 1:2].values
    actual_price=data_array.iloc[-29:, 1:2].values
    xtest=data_array.iloc[-30:-1,[1,6,7,8,9,12,13]]
    
    from sklearn.preprocessing import StandardScaler
    scaling=StandardScaler()
    xtrain=scaling.fit_transform(xtrain)
    xtest=scaling.transform(xtest)
    ytrain=scaling.fit_transform(ytrain)
    from sklearn.linear_model import LinearRegression
    linear_regression=LinearRegression()
    linear_regression.fit(xtrain, ytrain)
    
    y_prediction = linear_regression.predict(xtest)
    y_prediction=scaling.inverse_transform(y_prediction)
    plt.plot(actual_price,label='Stock Price-actual',color='black')
    plt.plot(y_prediction,label='Stock Price trend predicted',color='green')
    heading=str(name[0]+' Stock Price Prediction')
    plt.title(heading)
    plt.xlabel('Time axis')
    yaxis=str(name[0]+' Stock Price')
    plt.ylabel(yaxis)
    plt.legend()
    plt.show()
    print(mean_squared_error(actual_price, y_prediction))
    
    
# Creating  a function to predict the stock price for selected stocks
def predictPriceLSTM(stockname):
    name=stockname.split('_')
    print('Stock Analyzed:',name[0])
    data_array=pd.read_excel(stockname)
    data_array=data_array.dropna()
    #sorting by date
    data_array=data_array.sort_values(by=['Date'])
    training_data = data_array.iloc[:-31, 1:2].values
    
    #Scaling the data
    from sklearn.preprocessing import MinMaxScaler
    scaling=MinMaxScaler(feature_range=(0, 1))
    scaled_data=scaling.fit_transform(training_data)
    
    xtrain,ytrain,xtest=[],[],[]
    # Using 50 steps
    for x in range(50, len(training_data)):
        xtrain.append(scaled_data[x-50:x, 0])
        ytrain.append(scaled_data[x,0])
    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain)
    xtrain=np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1], 1))
    
    #Creating the model
    lstmregression = Sequential()
    #Add the LSTM layers and Dropout 
    lstmregression.add(LSTM(units=70,input_shape=(xtrain.shape[1],1),return_sequences=True))
    lstmregression.add(Dropout(0.2))
    lstmregression.add(LSTM(units=70,return_sequences=True))
    lstmregression.add(Dropout(0.1))
    lstmregression.add(LSTM(units=70,return_sequences=True))
    lstmregression.add(Dropout(0.2))
    lstmregression.add(LSTM(units=70))
    lstmregression.add(Dropout(0.2))
    lstmregression.add(Dense(units=1))
    lstmregression.compile(loss='mean_squared_error',optimizer='adam')
    # Apply model on training set
    lstmregression.fit(xtrain,ytrain,batch_size=32,epochs=10)
    
    actual_price=data_array.iloc[-31:, 1:2].values
    # Prediction
    all_data= data_array['Last Price']
    test_input=all_data[len(all_data)-len(actual_price)-50:].values
    test_input=test_input.reshape(-1,1)
    test_input=scaling.transform(test_input)
    
    for i in range(50, 81):
        xtest.append(test_input[i-50:i, 0])
    xtest=np.array(xtest)
    xtest=np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))
    prediction=lstmregression.predict(xtest)
    # getting prices
    prediction=scaling.inverse_transform(prediction)
    # Plotting chart for actual and prediction
    plt.plot(actual_price,label='Stock Price-actual',color='black')
    plt.plot(prediction,label='Stock Price-predicted',color='green')
    heading=str(name[0]+' Stock Price Prediction')
    plt.title(heading)
    plt.xlabel('Time axis')
    yaxis=str(name[0]+' Stock Price')
    plt.ylabel(yaxis)
    plt.legend()
    plt.show()
    print(mean_squared_error(actual_price, prediction))
    print('End----------------------------------------------------')



if __name__ == '__main__': 
    Apple='AAPL_5_year_data.xlsx'
    IBM='IBM_5_year_data.xlsx'
    Ford='FORD_5_year_data.xlsx'
    GE='GE_5_year_data.xlsx'
    FB='FB_5_year_data.xlsx'
    #Creating a list of stocks  Apple,IBM,Ford,GE,FB
    Stocks=[Apple,IBM,Ford,GE,FB]
    
    for stock in Stocks:
        #Predict stock price using LSTM
        predictPriceLSTM(stock)
        #Predict stock price using Multiple linear regression
        predictPriceLR(stock)