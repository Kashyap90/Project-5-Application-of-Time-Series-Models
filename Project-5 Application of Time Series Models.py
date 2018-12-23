
# coding: utf-8

# In[1]:


# #Problem Statement:
#Pick up the following stocks and generate forecasts accordingly
# Stocks:

#1. NASDAQ.AAPL

#2. NASDAQ.ADP

#3. NASDAQ.CBOE

#4. NASDAQ.CSCO

#5. NASDAQ.EBAY

#Dataset Link: https://drive.google.com/file/d/1VxoJDgyiAdMRI7-Fp7RxazDTvQ9Lw54d/


# In[2]:


# Importing Modules:


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA, ARMAResults
import datetime
import sys
import seaborn as sns
import statsmodels
import statsmodels.stats.diagnostic as diag
from statsmodels.tsa.stattools import adfuller
from scipy.stats.mstats import normaltest
from matplotlib.pyplot import acorr
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


# In[4]:


# Loading Data:


# In[5]:


#Read CSV (comma-separated) file into DataFrame:

df = pd.read_csv('C:/Users/kashyap/Downloads/data_stocks.csv')


# In[6]:


# Data Exploration/Analysis:


# In[7]:


#Returns the first 5 rows of df dataframe:

df.head()


# In[8]:


#The summary statistics of the df dataframe:


# In[9]:


df.describe()


# In[10]:


#Prints information about df DataFrame:


# In[11]:


df.info()


# In[12]:


#Columns of df dataframe:


# In[13]:


df.columns


# In[14]:


#Return a tuple representing the dimensionality of df DataFrame:


# In[15]:


df.shape


# In[16]:


#Check for any NA’s in the dataframe:


# In[17]:


df.isnull().values.any()


# In[18]:


#1. NASDAQ.AAPL:


# In[19]:


#Makes a copy of df dataframe:


# In[20]:


df_1 = df.copy()


# In[21]:


#Creating a column 'AAPL_LOG' with the log values of 'NASDAQ.AAPL' column data:

df_1["AAPL_LOG"] = df_1["NASDAQ.AAPL"].apply(lambda x:np.log(x))


# In[22]:


#Returns the first 5 rows of df_1 dataframe:


# In[23]:


df_1.head()


# In[24]:


#Type of the 'DATE' column:


# In[25]:


type(df_1["DATE"][0])


# In[26]:


#Creating a new column 'DATE_NEW' with formatted timestamp:

df_1["DATE_NEW"] = df_1["DATE"].apply(lambda x:dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))


# In[27]:


#Returns the first 5 rows of df_1 dataframe:


# In[28]:


df_1.head()


# In[29]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson 
#values above 2. 
#Prints Durbin-Watson statistic of given data.
print("Durbin-Watson statistic:",sm.stats.durbin_watson(df_1["AAPL_LOG"]))


# In[30]:


#Series Plot:

df_1["AAPL_LOG"].plot(figsize=(16,9))
plt.show()


# In[31]:


#Autocorrelation Plot
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_1["AAPL_LOG"].values.squeeze(), lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_1["AAPL_LOG"], lags=35, ax=ax2)


# In[32]:


#Getting the 'AAPL_LOG' column values as array with dropping NaN values:

array_1 = (df_1["AAPL_LOG"].dropna().as_matrix())


# In[33]:


#Creating a column 'AAPL_LOG_DIFF' with data as difference of 'AAPL_LOG' column current row and previous row:

df_1["AAPL_LOG_DIFF"] = df_1["AAPL_LOG"] - df_1["AAPL_LOG"].shift(periods=-1)


# In[34]:


#Prints model parameter:


# In[35]:


#Creating ARMA Model:

model_1 = sm.tsa.ARMA(array_1,(2,0)).fit()
print(model_1.params)


# In[36]:


#Printing Model's AIC, BIC and HQIC values:

print(model_1.aic, model_1.bic, model_1.hqic)


# In[37]:


#Finding the best values for ARIMA model parameter
aic=999999
a,b,c = 0,0,0

for p in range(3):
    for q in range(1,3):
        for r in range(3):
            try:
                model= ARIMA(array_1,(p,q,r)).fit()
                if(aic > model_1.aic):
                    aic = model_1.aic
                    a,b,c = p,q,r
            except:
                pass
                
print(a,b,c)


# In[38]:


#Creating and fitting ARIMA model:

model_1_arima = ARIMA(array_1,(0, 1, 0)).fit()


# In[39]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson 
#values above 2. 
#Prints Durbin-Watson statistic of given data.

print("Durbin-Watson statistic:",sm.stats.durbin_watson(model_1_arima.resid))


# In[40]:


#Predicting the values using ARIMA Model:

pred_1 = model_1_arima.predict()
pred_1


# In[41]:


# Root Mean Squared Error:


# In[42]:


#Printing RMSE value for the model:

print(np.sqrt(mean_squared_error(pred_1,df_1["AAPL_LOG_DIFF"][:-1])))


# In[43]:


#2. NASDAQ.ADP:


# In[44]:


#Makes a copy of df dataframe:


# In[45]:


df_2 = df.copy()


# In[46]:


#Creating a column 'ADP_LOG' with the log values of 'NASDAQ.ADP' column data:

df_2["ADP_LOG"] = df_2["NASDAQ.ADP"].apply(lambda x:np.log(x))


# In[47]:


#Returns the first 5 rows of df_2 dataframe:


# In[48]:


df_2.head()


# In[49]:


#Type of the 'DATE' column:

type(df_2["DATE"][0])


# In[50]:


#Creating a new column 'DATE_NEW' with formatted timestamp:

df_2["DATE_NEW"] = df_2["DATE"].apply(lambda x:dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))


# In[51]:


#Returns the first 5 rows of df_2 dataframe:


# In[52]:


df_2.head()


# In[53]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson
#values above 2. 
#Prints Durbin-Watson statistic of given data.

print("Durbin-Watson statistic:",sm.stats.durbin_watson(df_2["ADP_LOG"]))


# In[54]:


#Series Plot:

df_2["ADP_LOG"].plot(figsize=(16,9))
plt.show()


# In[55]:


#Autocorrelation Plot:

fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_2["ADP_LOG"].values.squeeze(), lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_2["ADP_LOG"], lags=35, ax=ax2)


# In[56]:


#Getting the 'AAPL_LOG' column values as array with dropping NaN values:

array_2 = (df_2["ADP_LOG"].dropna().as_matrix())


# In[57]:


#Creating a column 'ADP_LOG_DIFF' with data as difference of 'ADP_LOG' column current row and previous row:

df_2["ADP_LOG_DIFF"] = df_2["ADP_LOG"] - df_2["ADP_LOG"].shift(periods=-1)


# In[59]:


#Prints model parameter:


# In[60]:


#Creating ARMA Model:

model_2 = sm.tsa.ARMA(array_2,(2,0)).fit()
print(model_2.params)


# In[61]:


#Printing Model's AIC, BIC and HQIC values:

print(model_2.aic, model_2.bic, model_2.hqic)


# In[62]:


#Finding the best values for ARIMA model parameter:

aic=999999
a,b,c = 0,0,0

for p in range(3):
    for q in range(1,3):
        for r in range(3):
            try:
                model= ARIMA(array_2,(p,q,r)).fit()
                if(aic > model_2.aic):
                    aic = model_2.aic
                    a,b,c = p,q,r
            except:
                pass
                
print(a,b,c)


# In[63]:


#Creating and fitting ARIMA model:

model_2_arima = ARIMA(array_2,(0, 1, 0)).fit()


# In[64]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with 
#Durbin-Watson values above 2. 
#Prints Durbin-Watson statistic of given data.

print("Durbin-Watson statistic:",sm.stats.durbin_watson(model_2_arima.resid))


# In[65]:


#Predicting the values using ARIMA Model:

pred_2 = model_2_arima.predict()
pred_2


# In[66]:


#Root Mean Squared Error:


# In[67]:


#Printing RMSE value for the model:

print(np.sqrt(mean_squared_error(pred_2,df_2["ADP_LOG_DIFF"][:-1])))


# In[68]:


#3. NASDAQ.CBOE


# In[69]:


#Makes a copy of df dataframe:


# In[70]:


df_3 = df.copy()


# In[71]:


#Creating a column 'CBOE_LOG' with the log values of 'NASDAQ.CBOE' column data:

df_3["CBOE_LOG"] = df_3["NASDAQ.CBOE"].apply(lambda x:np.log(x))


# In[72]:


#Returns the first 5 rows of df_3 dataframe:


# In[73]:


df_3.head()


# In[74]:


#Type of the 'DATE' column:

type(df_3["DATE"][0])


# In[75]:


#Creating a new column 'DATE_NEW' with formatted timestamp:

df_3["DATE_NEW"] = df_3["DATE"].apply(lambda x:dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))


# In[76]:


#Returns the first 5 rows of df_3 dataframe:

df_3.head()


# In[77]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson 
#values above 2. 
#Prints Durbin-Watson statistic of given data.

print("Durbin-Watson statistic:",sm.stats.durbin_watson(df_3["CBOE_LOG"]))


# In[78]:


#Series Plot:

df_3["CBOE_LOG"].plot(figsize=(16,9))
plt.show()


# In[79]:


#Autocorrelation Plot:

fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_3["CBOE_LOG"].values.squeeze(), lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_3["CBOE_LOG"], lags=35, ax=ax2)


# In[80]:


#Getting the 'CBOE_LOG' column values as array with dropping NaN values:

array_3 = (df_3["CBOE_LOG"].dropna().as_matrix())


# In[81]:


#Creating a column 'CBOE_LOG_DIFF' with data as difference of 'CBOE_LOG' column current row and previous row:

df_3["CBOE_LOG_DIFF"] = df_3["CBOE_LOG"] - df_3["CBOE_LOG"].shift(periods=-1)


# In[82]:


#Prints model parameter:


# In[83]:


#Creating ARMA Model:

model_3 = sm.tsa.ARMA(array_3,(2,0)).fit()
print(model_3.params)


# In[84]:


#Printing Model's AIC, BIC and HQIC values:

print(model_3.aic, model_3.bic, model_3.hqic)


# In[85]:


#Finding the best values for ARIMA model parameter
aic=999999
a,b,c = 0,0,0

for p in range(3):
    for q in range(1,3):
        for r in range(3):
            try:
                model= ARIMA(array_3,(p,q,r)).fit()
                if(aic > model_3.aic):
                    aic = model_3.aic
                    a,b,c = p,q,r
            except:
                pass
                
print(a,b,c)


# In[86]:


#Creating and fitting ARIMA model:

model_3_arima = ARIMA(array_3,(0, 1, 0)).fit()


# In[87]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson 
#values above 2. 
#Prints Durbin-Watson statistic of given data.

print("Durbin-Watson statistic:",sm.stats.durbin_watson(model_3_arima.resid))


# In[88]:


#Predicting the values using ARIMA Model:

pred_3 = model_3_arima.predict()
pred_3


# In[89]:


#Root Mean Squared Error:


# In[90]:


#Printing RMSE value for the model:

print(np.sqrt(mean_squared_error(pred_3,df_3["CBOE_LOG_DIFF"][:-1])))


# In[91]:


# 4. NASDAQ.CSCO


# In[92]:


#Makes a copy of df dataframe:


# In[93]:


df_4 = df.copy()


# In[94]:


#Creating a column 'CSCO_LOG' with the log values of 'NASDAQ.CSCO' column data:

df_4["CSCO_LOG"] = df_4["NASDAQ.CSCO"].apply(lambda x:np.log(x))


# In[95]:


#Returns the first 5 rows of df_4 dataframe:


# In[96]:


df_4.head()


# In[97]:


#Type of the 'DATE' column:

type(df_4["DATE"][0])


# In[98]:


#Creating a new column 'DATE_NEW' with formatted timestamp:

df_4["DATE_NEW"] = df_4["DATE"].apply(lambda x:dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))


# In[99]:


#Returns the first 5 rows of df_4 dataframe:


# In[100]:


df_4.head()


# In[102]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson 
#values above 2. 
#Prints Durbin-Watson statistic of given data.

print("Durbin-Watson statistic:",sm.stats.durbin_watson(df_4["CSCO_LOG"]))


# In[103]:


#Series Plot:

df_4["CSCO_LOG"].plot(figsize=(16,9))
plt.show()


# In[104]:


#Autocorrelation Plot:

fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_4["CSCO_LOG"].values.squeeze(), lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_4["CSCO_LOG"], lags=35, ax=ax2)


# In[105]:


#Getting the 'CSCO_LOG' column values as array with dropping NaN values:

array_4 = (df_4["CSCO_LOG"].dropna().as_matrix())


# In[106]:


#Creating a column 'AAPL_LOG_DIFF' with data as difference of 'AAPL_LOG' column current row and previous row:

df_4["CSCO_LOG_DIFF"] = df_4["CSCO_LOG"] - df_4["CSCO_LOG"].shift(periods=-1)


# In[107]:


#Prints model parameter:


# In[108]:


#Creating ARMA Model:

model_4 = sm.tsa.ARMA(array_4,(2,0)).fit()
print(model_4.params)


# In[109]:


#Printing Model's AIC, BIC and HQIC values:

print(model_4.aic, model_4.bic, model_4.hqic)


# In[110]:


#Finding the best values for ARIMA model parameter:

aic=999999
a,b,c = 0,0,0

for p in range(3):
    for q in range(1,3):
        for r in range(3):
            try:
                model= ARIMA(array_4,(p,q,r)).fit()
                if(aic > model_4.aic):
                    aic = model_4.aic
                    a,b,c = p,q,r
            except:
                pass
                
print(a,b,c)


# In[111]:


#Creating and fitting ARIMA model:

model_4_arima = ARIMA(array_4,(0, 1, 0)).fit()


# In[112]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson 
#values above 2. 
#Prints Durbin-Watson statistic of given data.

print("Durbin-Watson statistic:",sm.stats.durbin_watson(model_4_arima.resid))


# In[113]:


#Predicting the values using ARIMA Model:

pred_4 = model_4_arima.predict()
pred_4


# In[114]:


# Root Mean Squared Error:


# In[115]:


#Printing RMSE value for the model:

print(np.sqrt(mean_squared_error(pred_4,df_4["CSCO_LOG_DIFF"][:-1])))


# In[116]:


# 5. NASDAQ.EBAY


# In[117]:


#Makes a copy of df dataframe:


# In[118]:


df_5 = df.copy()


# In[119]:


#Creating a column 'EBAY_LOG' with the log values of 'NASDAQ.EBAY' column data:

df_5["EBAY_LOG"] = df_5["NASDAQ.EBAY"].apply(lambda x:np.log(x))


# In[120]:


#Returns the first 5 rows of df_5 dataframe:


# In[121]:


df_5.head()


# In[122]:


#Type of the 'DATE' column:

type(df_5["DATE"][0])


# In[123]:


#Creating a new column 'DATE_NEW' with formatted timestamp:

df_5["DATE_NEW"] = df_5["DATE"].apply(lambda x:dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))


# In[124]:


#Returns the first 5 rows of df_5 dataframe:


# In[125]:


df_5.head()


# In[126]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson 
#values above 2. 
#Prints Durbin-Watson statistic of given data.

print("Durbin-Watson statistic:",sm.stats.durbin_watson(df_5["EBAY_LOG"]))


# In[127]:


#Series Plot:

df_5["EBAY_LOG"].plot(figsize=(16,9))
plt.show()


# In[128]:


#Autocorrelation Plot:

fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_5["EBAY_LOG"].values.squeeze(), lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_5["EBAY_LOG"], lags=35, ax=ax2)


# In[129]:


#Getting the 'EBAY_LOG' column values as array with dropping NaN values:

array_5 = (df_5["EBAY_LOG"].dropna().as_matrix())


# In[130]:


#Creating a column 'EBAY_LOG_DIFF' with data as difference of 'EBAY_LOG' column row and previous row:

df_5["EBAY_LOG_DIFF"] = df_5["EBAY_LOG"] - df_5["EBAY_LOG"].shift(periods=-1)


# In[131]:


#Prints model parameter:


# In[133]:


#Creating ARMA Model:

model_5 = sm.tsa.ARMA(array_5,(2,0)).fit()
print(model_5.params)


# In[134]:


#Printing Model's AIC, BIC and HQIC values:

print(model_5.aic, model_5.bic, model_5.hqic)


# In[135]:


#Finding the best values for ARIMA model parameter:

aic=999999
a,b,c = 0,0,0

for p in range(3):
    for q in range(1,3):
        for r in range(3):
            try:
                model= ARIMA(array_5,(p,q,r)).fit()
                if(aic > model_5.aic):
                    aic = model_5.aic
                    a,b,c = p,q,r
            except:
                pass
                
print(a,b,c)


# In[136]:


#Creating and fitting ARIMA model:

model_5_arima = ARIMA(array_5,(0, 1, 0)).fit()


# In[137]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson 
#values above 2. 
#Prints Durbin-Watson statistic of given data.

print("Durbin-Watson statistic:",sm.stats.durbin_watson(model_5_arima.resid))


# In[138]:


#Predicting the values using ARIMA Model:

pred_5 = model_5_arima.predict()
pred_5


# In[139]:


# Root Mean Squared Error:


# In[140]:


#Printing RMSE value for the model:

print(np.sqrt(mean_squared_error(pred_5,df_5["EBAY_LOG_DIFF"][:-1])))

