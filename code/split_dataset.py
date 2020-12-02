import pandas as pd
from forecast import *

def load_rename(path):
    """
    take in a path/to/csv,
    rename columns and make all lowercase column names,
    and return a dataframe
    """
    df = pd.read_csv(path)
    cols = get_col_names(df)
    # {'ACTIVITY DATE':'date', 
    #                 'Sum of TOTAL TIME - AHT (sec)': 'handle_time',
    #                 'Sum of TOTAL TIME - AHT - FORECAST (sec)': 'handle_time_forecast', 
    #                 'Sum of VOLUME - RECEIVED': 'volume', 
    #                 'Sum of VOLUME - FORECAST': 'volume_forecast' }
    df.rename(columns=cols, 
            inplace=True)
    # make all col names lower case
    df.columns = [x.lower() for x in df.columns]
    print(df)
    return df

def write_data(df, group='business line', 
forecast_list=['handle_time','handle_time_forecast', 
'volume', 'volume_forecast']):
    """
    take in a dataframe, query by business unit 
    and write to a csv
    """
    # iterate through the unique business lines
    groups = df[group].unique()
    for line in groups:
        data = df[df[group] == line][['date']+forecast_list]
        path = './data/'+line+'_aht_vol.csv'
        path = path.replace(' ', '-')
        data.to_csv(path)
        print('wrote to '+path)

# load a dataframe
df = load_rename('./data/full_dataset.csv')

# create csv for individual business unit
write_data(df)