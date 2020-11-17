
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import sys
import re

def plot_time_vol(df):
    plt.subplot(221)
    df['handle_time'].plot.box()

    plt.subplot(222)
    df['volume'].plot.box()

    plt.subplot(223)
    df['handle_time'].hist(grid=False)

    plt.subplot(224)
    df['volume'].hist(grid=False)
    plt.tight_layout()
    plt.show()


def create_training_data(df, kpi, start_date, end_date):
    # create a new dataframe with 2 columns, date and actual volume received
    df['date'] = df.index
    df = df[['date', kpi]]

    # Must pre-format column names
    df.columns = ['ds','y']

    # set start date of training data
    df = df[df['ds'] >= start_date]

    # set end date of training data
    df = df[df['ds'] < end_date]

    return df

def create_forecast(df,periods=200):
    # create default prophet model
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(df)

    future = m.make_future_dataframe(periods=periods)
    # display(future.tail())

    # get rid of holidays and weekends
    future = future[(~future['ds'].isin(h.iloc[:,0].tolist())) & (~future['ds'].dt.weekday.isin([5,6]))]

    return m,future

def plot_forecast(forecast):
    # view only the last 5 predictions with confidence intervals 
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    fig1 = m.plot(forecast)

    fig2 = m.plot_components(forecast)

def set_forecast(forecast, df1, start_date, end_date):
    # set forecast beginning and end date
    f = forecast.copy()
    f = f[(forecast['ds'] >= start_date) & (f['ds'] <= end_date)]

    # create validation dataset
    df2 = df1.copy()
    df2['ds'] = pd.to_datetime(df2.index.date) 

    # set start date of validation data equal to June 1st, 2020
    df2 = df2[df2['ds'] >= start_date]

    # remove weekends and holidays
    df2 = df2[~df2.index.isin(h.iloc[:,0].tolist())]
    df2 = df2[~df2.index.weekday.isin([5,6])]
    
    return f, df2

def validate_dates(f, df2):
    """
    take in forecast, f
    and dataframe df2 and return Bool
    """
    res = []
    # validate forecast dates shape equals actual dates shape
    v1 = df2.shape[0] == f.shape[0]
    if not v1:
        return False
    else: res.append(v1)
        
    # validate all dates for forecast equal actual dates
    res.append(all(df2['ds'].dt.date.values == f['ds'].dt.date.values))
    
    return all(res)

def evaluate_model(forecast, kpi, metric='mae'):
    if metric == 'mae':
        mae1 = mean_absolute_error(df2[kpi], df2[kpi+'_forecast'])
        mae2 = mean_absolute_error(df2[kpi], f['yhat'])
        diff = mae1- mae2
        return {'old': mae1, 'prophet': mae2}
    else:
        return 'invalid metric'

def validate_holidays(df):
    # validate no weekends or holidays
    return (df.index.weekday.isin([5,6]).any(),
    df.index.isin(h.iloc[:,0].tolist()).any()) == (False, False)

## Start script
# Load Data
def get_col_names(df):
    res = {}
    for i, col in enumerate(df.columns):

        p = re.compile(r'forecast', re.IGNORECASE) 
        q = re.compile(r'time', re.IGNORECASE)
        r = re.compile(r'volume', re.IGNORECASE)
        d = re.compile(r'date', re.IGNORECASE)
        fcst = p.search(col); time = q.search(col); vol = r.search(col); date = d.search(col)

        if fcst:
            if time:
                res[col] = 'handle_time_forecast'
            elif vol:
                res[col] = 'volume_forecast'
        elif time:
            res[col] = 'handle_time'
        elif vol:
            res[col] = 'volume'
        else:
            res[col] = 'date'
            
    return res

def remove_outliers(df, bu):
    if bu == 'ris':
        df = df[df['handle_time'] > 100000]
    elif bu == 'scs':
        df = df[(df['handle_time'] > 500000) & (df['aht'] > 450)]
    elif bu == 'wis':
        df = df[(df['handle_time'] > 2000000) & (df['volume'] < 10000)]
    elif bu == 'bro':
        df = df[df['handle_time'] > 100000]
    elif bu == 'psg':
        df = df[(df['handle_time'] > 100000)]
    elif bu == 'col':
        df = df[(df['handle_time'] > 100000)]
    return df

# df = pd.read_csv(file_name, parse_dates=['date'],
#                 index_col=['date'], usecols=range(5), header=1,
#                 names=['date', 'handle_time', 'handle_time_forecast', 
#                        'volume','volume_forecast'])

file_name, kpi, start_train, end_train, periods = (sys.argv[1], 
sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

df = pd.read_csv('data/'+file_name)
h = pd.read_csv('data/holidays.csv')
col_names = get_col_names(df)
df.rename(columns=col_names, inplace=True)

df['date'] = pd.to_datetime(df['date'])
df.index = df['date']

# create column for aht
df['aht'] = df['handle_time']/df['volume']

df['aht_forecast'] = df['handle_time_forecast']/df['volume_forecast']

df1 = df.copy()


# remove outliers
bu = file_name[:3]
df = remove_outliers(df, bu)

# remove holidays
df = df[~df.index.isin(h.iloc[:,0].tolist())]

# remove weekends
df = df[~df.index.weekday.isin([5,6])]
# print(validate_holidays(df))

# plot data
# plot_time_vol(df)

# create training
df = create_training_data(df, kpi, start_train, end_train)

# plot forecast
# display(df.head())
m,future = create_forecast(df, periods=int(periods))
forecast = m.predict(future)
# plot_forecast(forecast)

# set forecast beginning and end date
f = forecast.copy()
f = f[(forecast['ds'] >= end_train) & (f['ds'] <= df1.index.max())]

# create validation dataset
df2 = df1.copy()
df2['ds'] = pd.to_datetime(df2.index.date) 

# set start date of validation data equal to June 1st, 2020
df2 = df2[df2['ds'] >= end_train]

# remove weekends and holidays
df2 = df2[~df2.index.isin(h.iloc[:,0].tolist())]
df2 = df2[~df2.index.weekday.isin([5,6])]

# Validate test data
if validate_dates(f, df2):
    mae = evaluate_model(f, kpi, metric='mae')
    print('\nMean Absolute Error')
    print(mae)
#     display(pd.DataFrame(index=['old model', 'prophet', 'difference'], 
#                  data=[int(mae1), int(mae2), int(mae1-mae2)],
#                  columns=['MAE (volume)']))
else:
    print('invalid dates')

df_out = forecast[['ds', 'yhat_lower', 'yhat', 'yhat_upper']]
# df_out.to_csv('\data\preds'+'_'+file_name[:3] +'_'+kpi+'_'+start_train+'_'+end_train+'.csv', index=False)

print('division:', file_name[:3], 
'kpi:',kpi, 'training starting on ', 
start_train, 
'ending on ',
end_train)
# print(df1.max())
# print(df_out)