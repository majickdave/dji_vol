
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

# def get_eval(f,df2,kpi):
#     if validate_dates(f, df2):
#         mae = evaluate_model(f,df2, kpi, metric='mae')
#     else:
#         return 'invalid dates'
#     return mae

def get_pred_score(file_name, kpi, start_train, end_train, periods=365):
    """
    provide system arguments file_name, kpi, start_train, end_train, periods
    write top scores to ./scores/ and predictions to ./preds/ folders
    """
    # set names for columns in order
    names = ['date', 'handle_time', 'handle_time_forecast',
    'volume', 'volume_forecast']
    df = pd.read_csv('./data/'+file_name, names=names, header=1, index_col=0)

    df['date'] = pd.to_datetime(df['date'])
    df.index = df['date']

    # create column for aht
    df['aht'] = df['handle_time']/df['volume']
    df['aht_forecast'] = df['handle_time_forecast']/df['volume_forecast']

    # df = df[(df['aht'] > 200) & (df['aht'] < 1500)]

    df1 = df.copy()

    # create business unit name
    bu = file_name[:6]

    # # remove outliers if it helps the model
    # df = remove_outliers(df, bu)

    # remove holidays
    df = df[~df.index.isin(h.iloc[:,0].tolist())]

    # remove weekends
    df = df[~df.index.weekday.isin([5,6])]

    # # plot data in a notebook
    # plot data
    # plot_time_vol(df)

    # create training
    df = create_training_data(df, kpi, start_train, end_train)

    
    # display(df.head())
    m,future = create_forecast(df, periods=int(periods))
    forecast = m.predict(future)

    # # plot forecast in a notebook
    # plot_forecast(forecast)

    # set forecast beginning and end date
    f = forecast.copy()
    end_test = df1.index.max()
    f = f[(forecast['ds'] > end_train) & (f['ds'] <= end_test)]

    # create validation dataset
    df2 = df1.copy()
    df2['ds'] = pd.to_datetime(df2.index.date) 

    # set start date of validation data equal to June 1st, 2020 or any other date
    df2 = df2[df2['ds'] > end_train]

    # remove weekends and holidays from data
    df2 = df2[~df2.index.isin(h.iloc[:,0].tolist())]
    df2 = df2[~df2.index.weekday.isin([5,6])]
    os.makedirs('./preds/'+kpi, exist_ok=True)

    future_forecast=forecast[['ds', 'yhat_lower', 'yhat', 'yhat_upper']][forecast['ds']>datetime.datetime.now()]

    f[['ds', 'yhat_lower', 'yhat', 'yhat_upper']].to_csv('./preds/'+kpi+'/'+bu+'.csv')
    forecast.to_csv('./preds/analysis/'+kpi+'_'+bu+'.csv')

    # Validate test data, it must match for scoring
    mae = evaluate_model(f,df2, kpi, metric='mae')

    mae.update({'kpi': kpi, 'start_train':start_train, 'end_train': end_train, 'end_test': end_test}) 

    curr = pd.read_csv('scores/'+kpi+'_score.csv',index_col=0)
    new = pd.DataFrame(mae, index=[bu])

    new.to_csv('./scores/current/'+bu+'_'+kpi+'.csv')
    
    if bu not in curr.index:
        curr = pd.concat([curr, new], 0)
    # create log everytime a score is superceded
    elif new.loc[bu,'prophet'] < curr.loc[bu,'prophet']:
        data = pd.concat([curr.loc[[bu],:],new.loc[[bu],:]],0)
        data.to_csv('./scores/logs/'+bu+'_'+kpi+'_'+datetime.datetime.now()
        .strftime("%b %d %Y %H:%M:%S").replace(' ', '_')+'.csv')
        curr.update(new)
        curr.to_csv('./scores/'+kpi+'_score.csv')

        return 'error with metric'
    print('\ntraining',file_name, 'on', kpi, 'start train:',start_train, 
    'end train:', end_train, 'end_test:', )


# def train_predict():
#     """
#     TODO
#     create an api
#     """
#     try:
#         if sys.argv[1]: 
#             file_name, kpi, start_train, end_train, periods = (sys.argv[1], 
#             sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
#             get_pred_score(file_name, kpi, start_train, end_train, periods=200)
#         else:
#             pass
#     except:
#         print("""
#         wrong number of arguments, try providing these 5 args:
#         file_name kpi start_train end_train periods
#         """)

# train_predict()

if __name__ == '__main__':
    try:
        file_name, kpi, start_train, end_train, periods = (sys.argv[1], 
        sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        get_pred_score(file_name, kpi, start_train, end_train, periods=200)
    except ValueError as e:
        print(e,"""\n
        wrong number of arguments, try providing these 5 args:
        file_name kpi start_train end_train periods
        """)



