
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import sys
import re
import os
import datetime
from forecast import *

h = pd.read_csv('./data/holidays.csv')

def get_pred_score(file_name, kpi, start_train, end_train, periods=365):
    """
    provide system arguments file_name, kpi, start_train, end_train, periods
    write top scores to ./scores/ and predsictions to ./preds/ folders
    """
    names = ['date', 'handle_time', 'handle_time_forecast',
    'volume', 'volume_forecast']
    df = pd.read_csv('./data/'+file_name, names=names, header=1, index_col=0)

    df['date'] = pd.to_datetime(df['date'])
    df.index = df['date']

    # create column for aht
    df['aht'] = df['handle_time']/df['volume']

    df['aht_forecast'] = df['handle_time_forecast']/df['volume_forecast']

    df1 = df.copy()

    # remove outliers
    bu = file_name[:6]
    # df = remove_outliers(df, bu)

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
    f = f[(forecast['ds'] > end_train) & (f['ds'] <= df1.index.max())]

    # create validation dataset
    df2 = df1.copy()
    df2['ds'] = pd.to_datetime(df2.index.date) 

    # set start date of validation data equal to June 1st, 2020
    df2 = df2[df2['ds'] >= end_train]

    # remove weekends and holidays
    df2 = df2[~df2.index.isin(h.iloc[:,0].tolist())]
    df2 = df2[~df2.index.weekday.isin([5,6])]
    if not os.path.exists('./preds/'+kpi):
        os.mkdir('./preds/'+kpi)
        return
    forecast[['ds', 'yhat_lower', 'yhat', 'yhat_upper']].to_csv('./preds/'+kpi+'/'+bu+'.csv')
    
    # Validate test data
    def get_eval():
        if validate_dates(f, df2):
            mae = evaluate_model(f,df2, kpi, metric='mae')
        else:
            return 'invalid dates'
        return mae

    mae = get_eval()

    try:
        mae.update({'kpi': kpi, 'start_train':start_train, 'end_train': end_train}) 
        curr = pd.read_csv('scores/'+kpi+'_score.csv',index_col=0)
        new = pd.DataFrame(mae, index=[bu])


        if bu not in curr.index:
            curr = pd.concat([curr, new], 0)
        # create log everytime a score is superceded
        elif new.loc[bu,'prophet'] < curr.loc[bu,'prophet']:
            data = pd.concat([curr.loc[[bu],:],new.loc[[bu],:]],0)
            data.to_csv('./scores/logs/'+bu+'_'+kpi+'_'+datetime.datetime.now()
            .strftime("%b %d %Y %H:%M:%S").replace(' ', '_')+'.csv')
            curr.update(new)
            curr.to_csv('scores/'+kpi+'_score.csv')
    except:
        return 'error with metric'
    print('\ntraining',file_name, 'on', kpi, 'start:',start_train, 'end:', end_train+'\n')
    print(curr.loc[bu,:])


# try:
#     if sys.argv[1]: 
#         file_name, kpi, start_train, end_train, periods = (sys.argv[1], 
#         sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
#         get_pred_score(file_name, kpi, start_train, end_train, periods=200)
#     else:
#         pass
# except:
#     print("""
#     wrong number of arguments, try providing these 5 args:
#     file_name kpi start_train end_train periods
#     """)